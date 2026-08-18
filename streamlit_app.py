# -*- coding: utf-8 -*-
"""
🏥 КЛИНИКА-АНАЛИЗАТОР — Streamlit Edition v1.0
Анализ удалённости пациентов от клиники с геокодированием и сегментацией.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Photon, Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from scipy.ndimage import gaussian_filter1d
import time
import json
import os
import re
import io
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
#  НАСТРОЙКИ СТРАНИЦЫ
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Клиника-Анализатор",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════
PHOTON_DELAY      = 0.35
NOMINATIM_DELAY   = 1.1
AUTOSAVE_EVERY    = 50
CACHE_FILENAME    = "geo_cache.json"

SEGMENT_ORDER = ["0–2 км", "2–5 км", "5–7 км", "7–10 км", "10+ км", "Другие города", "Нет данных"]
COLORS = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#95a5a6']

# ═══════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════

def init_session_state():
    """Инициализация session_state для хранения данных между ререндерами."""
    defaults = {
        'cache': {},
        'df_processed': None,
        'agg': None,
        'clinic_coord': None,
        'clinic_addr': None,
        'clinic_norm_city': None,
        'date_range_str': '',
        'processing_done': False,
        'col_map': {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def clean_part(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def normalize_city(city):
    """Убирает 'г', 'г.', 'п', 'пгт' и т.д. в конце для сравнения."""
    if pd.isna(city):
        return ""
    city = str(city).strip().lower()
    city = re.sub(r'\s+(г|г\.|п|п\.|с|с\.|пгт|пгт\.|рп|рп\.|пос|пос\.|село|дер|д|аул|кп|т|с/с|с/о|тер|ж/м|мкр|р-н|обл|край|респ)\.?$', '', city)
    return city.strip()

def build_address(house, street, city):
    city = clean_part(city)
    street = clean_part(street)
    house = clean_part(house)
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', house):
        house = ""
    parts = [p for p in [house, street, city, "Россия"] if p]
    return ", ".join(parts)

def find_col(cols, keys):
    for c in cols:
        c_lower = str(c).lower()
        for k in keys:
            if k in c_lower:
                return c
    return None

def assign_segment(meters):
    if pd.isna(meters):
        return "Нет данных"
    km = meters / 1000
    if km <= 2:
        return "0–2 км"
    elif km <= 5:
        return "2–5 км"
    elif km <= 7:
        return "5–7 км"
    elif km <= 10:
        return "7–10 км"
    else:
        return "10+ км"

def assign_segment_v8(row, clinic_norm_city):
    """Сначала проверяем город, потом расстояние."""
    if row['norm_city'] != clinic_norm_city:
        return "Другие города"
    return assign_segment(row['distance_m'])

def get_distance_meters(coord_a, coord_b):
    if coord_a is None or coord_b is None:
        return None
    return geodesic(coord_a, coord_b).meters

def geocode_with_fallback(geo_photon, geo_nominatim, address, cache):
    if address in cache and cache[address] is not None:
        return cache[address]
    try:
        loc = geo_photon.geocode(address, timeout=15)
        if loc:
            cache[address] = [loc.latitude, loc.longitude]
            time.sleep(PHOTON_DELAY)
            return cache[address]
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass
    try:
        time.sleep(0.5)
        loc = geo_nominatim.geocode(address, timeout=20)
        cache[address] = [loc.latitude, loc.longitude] if loc else None
        time.sleep(NOMINATIM_DELAY)
        return cache[address]
    except Exception:
        cache[address] = None
        time.sleep(NOMINATIM_DELAY)
        return None

# ═══════════════════════════════════════════════════════════════════════
#  UI: БОКОВАЯ ПАНЕЛЬ
# ═══════════════════════════════════════════════════════════════════════

def sidebar():
    st.sidebar.title("⚙️ Настройки")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 Адрес клиники")

    clinic_city = st.sidebar.text_input("Город клиники", value="", placeholder="Например: Москва")
    clinic_street = st.sidebar.text_input("Улица клиники", value="", placeholder="Например: Ленина")
    clinic_house = st.sidebar.text_input("Дом клиники", value="", placeholder="Например: 15")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Кэш геокодирования")

    cache_file = st.sidebar.file_uploader("Загрузить geo_cache.json (опционально)", type=["json"])
    if cache_file is not None:
        try:
            loaded_cache = json.load(cache_file)
            added = 0
            updated = 0
            for addr, coords in loaded_cache.items():
                # Умное слияние: не затираем существующие координаты пустыми значениями
                if addr not in st.session_state['cache']:
                    st.session_state['cache'][addr] = coords
                    added += 1
                elif st.session_state['cache'][addr] is None and coords is not None:
                    st.session_state['cache'][addr] = coords
                    updated += 1
            msg = f"✅ Кэш загружен: +{added} новых"
            if updated:
                msg += f", {updated} исправлено"
            st.sidebar.success(msg)
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки кэша: {e}")

    cache_size = len(st.session_state['cache'])
    st.sidebar.info(f"📍 Адресов в кэше: **{cache_size}**")

    return clinic_city, clinic_street, clinic_house

# ═══════════════════════════════════════════════════════════════════════
#  ОБРАБОТКА ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════

def extract_date_range(df_raw):
    """Ищем даты в первых 10 строках."""
    date_start = None
    date_end = None
    for i in range(min(10, len(df_raw))):
        row_text = ' '.join(str(x) for x in df_raw.iloc[i] if pd.notna(x))
        m_start = re.search(r'[СC]\s*:\s*(\d{2}\.\d{2}\.\d{4})', row_text)
        m_end = re.search(r'ПО\s*:\s*(\d{2}\.\d{2}\.\d{4})', row_text)
        if m_start:
            date_start = m_start.group(1)
        if m_end:
            date_end = m_end.group(1)

    if date_start and date_end:
        return f"Период анализа: {date_start} – {date_end}"
    return ""

def find_header_row(df_raw):
    """Автоопределение строки с заголовками."""
    for i in range(min(25, len(df_raw))):
        row_text = ' '.join(str(x).lower() for x in df_raw.iloc[i] if pd.notna(x))
        if ('город' in row_text) and ('улица' in row_text) and (('айди' in row_text) or ('дом' in row_text)):
            return i
    for i in range(min(25, len(df_raw))):
        row_text = ' '.join(str(x).lower() for x in df_raw.iloc[i] if pd.notna(x))
        if ('город' in row_text) and ('улица' in row_text):
            return i
    return None

def process_excel(uploaded_file, clinic_city, clinic_street, clinic_house):
    """Основной пайплайн обработки."""

    # --- Адрес клиники ---
    clinic_addr = build_address(clinic_house, clinic_street, clinic_city)
    clinic_norm_city = normalize_city(clinic_city)
    st.session_state['clinic_addr'] = clinic_addr
    st.session_state['clinic_norm_city'] = clinic_norm_city

    if not clinic_city or not clinic_street or not clinic_house:
        st.error("❌ Заполните все поля адреса клиники в боковой панели!")
        return False

    st.info(f"🔍 Адрес клиники: {clinic_addr}")

    # --- Чтение Excel ---
    xl = pd.ExcelFile(uploaded_file)
    sheet = 'Лист 2' if 'Лист 2' in xl.sheet_names else (xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0])
    st.write(f"📄 Работаем с листом: **{sheet}**")

    df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)

    # --- Даты ---
    date_range_str = extract_date_range(df_raw)
    st.session_state['date_range_str'] = date_range_str
    if date_range_str:
        st.success(f"📅 {date_range_str}")

    # --- Заголовки ---
    header_idx = find_header_row(df_raw)
    if header_idx is None:
        st.error("❌ Не удалось найти строку с заголовками (город, улица, дом).")
        return False

    st.write(f"🔍 Заголовки найдены в строке Excel №{header_idx + 1}")

    df = df_raw.iloc[header_idx + 1:].copy()
    df.columns = [str(c).strip().lower() for c in df_raw.iloc[header_idx]]
    df = df.reset_index(drop=True)
    df = df.dropna(how='all').reset_index(drop=True)

    st.write(f"📊 Всего строк данных: **{len(df)}**")

    # --- Поиск колонок ---
    col_city = find_col(df.columns, ['город'])
    col_street = find_col(df.columns, ['улица'])
    col_house = find_col(df.columns, ['дом'])
    col_serv = find_col(df.columns, ['усл', 'услуг'])
    col_sum = find_col(df.columns, ['сумма', 'оплат'])

    st.session_state['col_map'] = {
        'city': col_city, 'street': col_street, 'house': col_house,
        'serv': col_serv, 'sum': col_sum
    }

    if not col_city or not col_street or not col_house:
        st.error("❌ Не удалось найти обязательные колонки (город / улица / дом).")
        st.write(f"Найдены колонки: {list(df.columns)}")
        return False

    st.write(f"🔎 Найдены колонки: город=`{col_city}`, улица=`{col_street}`, дом=`{col_house}`")

    # Приводим к строкам
    df[col_city] = df[col_city].astype(str).replace('nan', '').replace('None', '')
    df[col_street] = df[col_street].astype(str).replace('nan', '').replace('None', '')
    df[col_house] = df[col_house].astype(str).replace('nan', '').replace('None', '')

    # --- Фильтр неполных адресов ---
    bad_mask = (
        (df[col_city].str.strip() == '') |
        (df[col_street].str.strip() == '') |
        (df[col_house].str.strip() == '') |
        (df[col_house].str.match(r'^\d{1,2}/\d{1,2}/\d{4}$'))
    )

    df['is_valid'] = ~bad_mask
    df.loc[bad_mask, 'segment'] = 'Нет данных'

    excluded = df[bad_mask].copy()
    df_good = df[~bad_mask].copy()

    df_good['norm_city'] = df_good[col_city].apply(normalize_city)

    st.write(f"🚫 Исключено (неполный адрес) → 'Нет данных': **{len(excluded)}** чел.")
    st.write(f"✅ Будет обработано: **{len(df_good)}** чел.")

    # --- Геокодирование ---
    cache = st.session_state['cache']

    geo_photon = Photon(user_agent="clinic_analyzer_streamlit", timeout=15)
    geo_nominatim = Nominatim(user_agent="clinic_analyzer_streamlit", timeout=20)

    # Клиника
    with st.spinner("🔍 Геокодирование клиники..."):
        if clinic_addr not in cache or cache[clinic_addr] is None:
            geocode_with_fallback(geo_photon, geo_nominatim, clinic_addr, cache)
        clinic_coord = cache.get(clinic_addr)

    if not clinic_coord:
        st.error("❌ Не удалось найти координаты клиники. Проверьте адрес.")
        return False

    st.session_state['clinic_coord'] = clinic_coord
    st.success(f"✅ Координаты клиники: {clinic_coord}")

    # Клиенты
    df_good['geo_address'] = df_good.apply(
        lambda r: build_address(r.get(col_house, ''), r.get(col_street, ''), r.get(col_city, '')),
        axis=1
    )
    df_good = df_good[df_good['geo_address'] != 'Россия'].copy()

    unique_addrs = [a for a in df_good['geo_address'].unique() if a not in cache]
    failed_before = sum(1 for a in df_good['geo_address'].unique() if a in cache and cache[a] is None)
    total_unique = len(df_good['geo_address'].unique())
    already_cached = total_unique - len(unique_addrs)

    st.write(f"📍 Уникальных адресов: **{total_unique}** | Уже в кэше: **{already_cached}** | Осталось: **{len(unique_addrs)}**")
    if failed_before:
        st.warning(f"⚠️ Ранее не удалось геокодировать (пропущены): {failed_before}")

    if unique_addrs:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, addr in enumerate(unique_addrs):
            geocode_with_fallback(geo_photon, geo_nominatim, addr, cache)
            progress = (i + 1) / len(unique_addrs)
            progress_bar.progress(min(progress, 0.99))
            status_text.text(f"Геокодирование: {i+1} / {len(unique_addrs)} — {addr[:50]}...")

        progress_bar.empty()
        status_text.empty()
        st.success("💾 Геокодирование завершено!")
    else:
        st.info("✅ Все адреса уже в кэше!")

    # --- Расчёт расстояний ---
    df_good['coords'] = df_good['geo_address'].map(lambda a: cache.get(a))
    df_good['distance_m'] = df_good['coords'].apply(lambda c: get_distance_meters(clinic_coord, c))
    df_good['distance_km'] = (df_good['distance_m'] / 1000).round(3)

    # --- Сегментация ---
    df_good['segment'] = df_good.apply(lambda r: assign_segment_v8(r, clinic_norm_city), axis=1)

    # Объединяем
    df = pd.concat([df_good, excluded], ignore_index=True)
    df['segment'] = df['segment'].fillna('Нет данных')

    df_ok = df.dropna(subset=['distance_m']).copy()
    st.write(f"📊 Успешно геокодировано: **{len(df_ok)}** из **{len(df_good)}** валидных")
    st.write(f"📊 Всего в отчёте: **{len(df)}** (включая **{len(excluded)}** без адреса)")

    # --- Агрегация ---
    if col_sum:
        df[col_sum] = pd.to_numeric(df[col_sum], errors='coerce').fillna(0)
    if col_serv:
        df[col_serv] = pd.to_numeric(df[col_serv], errors='coerce').fillna(0)

    agg = df.groupby('segment').agg(
        patients=('segment', 'count'),
        services=(col_serv, 'sum') if col_serv else ('segment', 'count'),
        revenue=(col_sum, 'sum') if col_sum else ('distance_m', 'sum'),
        avg_dist=('distance_m', 'mean')
    ).reindex(SEGMENT_ORDER).fillna(0)

    agg['patients_pct'] = (agg['patients'] / agg['patients'].sum() * 100).round(1)
    agg['revenue_pct'] = (agg['revenue'] / agg['revenue'].sum() * 100).round(1)

    st.session_state['df_processed'] = df
    st.session_state['agg'] = agg
    st.session_state['processing_done'] = True

    return True

# ═══════════════════════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════

def show_dashboard():
    df = st.session_state['df_processed']
    agg = st.session_state['agg']
    clinic_addr = st.session_state['clinic_addr']
    date_range_str = st.session_state['date_range_str']

    title_text = f'Анализ клиентов относительно клиники\n{clinic_addr}'
    if date_range_str:
        title_text += f'\n{date_range_str}'

    st.markdown("---")
    st.header("📊 Результаты анализа")

    # Таблица агрегации
    st.subheader("Сводка по сегментам")

    # Форматируем для отображения
    agg_display = agg.copy()
    agg_display['avg_dist'] = agg_display['avg_dist'].apply(lambda x: f"{x/1000:.1f} км" if x > 0 else "—")
    agg_display['revenue'] = agg_display['revenue'].apply(lambda x: f"{int(x):,} ₽".replace(",", " "))
    st.dataframe(agg_display, use_container_width=True)

    # Графики 2x2
    st.subheader("Графики")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title_text.replace('\n', '\n'), fontsize=14, fontweight='bold', y=1.02)

    # Количество
    ax1 = axes[0, 0]
    bars1 = ax1.bar(agg.index, agg['patients'], color=COLORS[:len(agg)], edgecolor='black')
    ax1.set_title('Количество пациентов', fontsize=12)
    ax1.set_ylabel('Человек')
    ax1.tick_params(axis='x', rotation=15)
    for i, (bar, val) in enumerate(zip(bars1, agg['patients'])):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(agg['patients'])*0.01,
                     f"{int(val)}\n({agg['patients_pct'].iloc[i]}%)", ha='center', va='bottom', fontsize=9)

    # Выручка
    ax2 = axes[0, 1]
    bars2 = ax2.bar(agg.index, agg['revenue'], color=COLORS[:len(agg)], edgecolor='black')
    ax2.set_title('Сумма оплат', fontsize=12)
    ax2.set_ylabel('Руб.')
    ax2.tick_params(axis='x', rotation=15)
    for i, (bar, val) in enumerate(zip(bars2, agg['revenue'])):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(agg['revenue'])*0.01,
                     f"{int(val):,}\n({agg['revenue_pct'].iloc[i]}%)", ha='center', va='bottom', fontsize=8)

    # Круговые
    ax3 = axes[1, 0]
    nonzero_patients = agg[agg['patients'] > 0]['patients']
    ax3.pie(nonzero_patients, labels=nonzero_patients.index, autopct='%1.1f%%', 
            colors=COLORS[:len(nonzero_patients)], startangle=140)
    ax3.set_title('Доля пациентов', fontsize=12)

    ax4 = axes[1, 1]
    nonzero_revenue = agg[agg['revenue'] > 0]['revenue']
    ax4.pie(nonzero_revenue, labels=nonzero_revenue.index, autopct='%1.1f%%', 
            colors=COLORS[:len(nonzero_revenue)], startangle=140)
    ax4.set_title('Доля выручки', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Плавный график распределения
    st.subheader("📈 Распределение по удалённости (только основной город)")

    df_good = df[df['segment'] != 'Другие города'].copy()
    df_good = df_good[df_good['segment'] != 'Нет данных'].copy()
    distances = df_good['distance_km'].dropna()
    distances_plot = distances[distances <= 50].copy()

    if len(distances_plot) > 0:
        fig2, ax = plt.subplots(figsize=(14, 6))

        x_max = distances_plot.quantile(0.995)
        if x_max < 5:
            x_max = 20

        bins = np.arange(0, x_max + 0.1, 0.1)
        counts, edges = np.histogram(distances_plot, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        counts_smooth = gaussian_filter1d(counts.astype(float), sigma=15)

        ax.plot(centers, counts_smooth, color='#2980b9', linewidth=2.5)
        ax.fill_between(centers, counts_smooth, alpha=0.25, color='#3498db')

        city_name = st.session_state['clinic_norm_city'].title()
        ax.set_title(f'Распределение по удалённости (только {city_name})', fontsize=13, fontweight='bold')
        ax.set_xlabel('Расстояние от клиники, км')
        ax.set_ylabel('Количество пациентов')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, max(counts_smooth) * 1.15)

        st.pyplot(fig2)
        plt.close(fig2)

        outliers = distances[distances > x_max]
        if len(outliers) > 0:
            st.caption(f"Ось X ограничена {x_max:.1f} км. Выбросов за пределами: {len(outliers)} чел.")
    else:
        st.info("Нет данных для построения графика распределения")

# ═══════════════════════════════════════════════════════════════════════
#  ЭКСПОРТ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════

def export_data():
    st.markdown("---")
    st.header("💾 Экспорт данных")

    df = st.session_state['df_processed']
    agg = st.session_state['agg']
    cache = st.session_state['cache']

    col1, col2, col3 = st.columns(3)

    with col1:
        # Детальный отчёт
        buffer_detail = io.BytesIO()
        with pd.ExcelWriter(buffer_detail, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Детальный')
        buffer_detail.seek(0)
        st.download_button(
            label="📥 Детальный отчёт (Excel)",
            data=buffer_detail,
            file_name=f"patients_distances_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col2:
        # Сводка
        buffer_summary = io.BytesIO()
        with pd.ExcelWriter(buffer_summary, engine='openpyxl') as writer:
            agg.to_excel(writer, sheet_name='Сводка')
        buffer_summary.seek(0)
        st.download_button(
            label="📊 Сводка по сегментам (Excel)",
            data=buffer_summary,
            file_name=f"segments_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col3:
        # Кэш
        cache_json = json.dumps(cache, ensure_ascii=False, indent=2)
        st.download_button(
            label="🗺 Кэш геокодирования (JSON)",
            data=cache_json,
            file_name="geo_cache.json",
            mime="application/json"
        )

    # Проверка по когорте
    st.markdown("---")
    st.subheader("🔍 Проверка данных по когорте")

    available_segments = [s for s in SEGMENT_ORDER if s in df['segment'].values]
    selected_segment = st.selectbox("Выберите сегмент для проверки:", available_segments)

    if selected_segment:
        cohort = df[df['segment'] == selected_segment].copy()
        st.write(f"📋 Сегмент **'{selected_segment}'**: {len(cohort)} записей")

        col_map = st.session_state['col_map']

        if selected_segment == 'Нет данных':
            display_cols = [col_map['city'], col_map['street'], col_map['house']]
            if col_map['sum']:
                display_cols.append(col_map['sum'])
        elif selected_segment == 'Другие города':
            display_cols = [col_map['city'], col_map['street'], col_map['house'], 'distance_km']
            if col_map['sum']:
                display_cols.append(col_map['sum'])
        else:
            display_cols = ['geo_address', 'distance_km']
            if col_map['sum']:
                display_cols.append(col_map['sum'])

        # Фильтруем только существующие колонки
        display_cols = [c for c in display_cols if c in cohort.columns]
        st.dataframe(cohort[display_cols].head(50), use_container_width=True)

        # Скачать когорту
        buffer_cohort = io.BytesIO()
        with pd.ExcelWriter(buffer_cohort, engine='openpyxl') as writer:
            cohort.to_excel(writer, index=False, sheet_name=selected_segment)
        buffer_cohort.seek(0)
        st.download_button(
            label=f"📥 Скачать когорту '{selected_segment}'",
            data=buffer_cohort,
            file_name=f"cohort_{selected_segment.replace(' ', '_').replace('–', '-').replace('+', 'plus')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ═══════════════════════════════════════════════════════════════════════
#  ГЛАВНЫЙ ЭКРАН
# ═══════════════════════════════════════════════════════════════════════

def main():
    init_session_state()

    st.title("🏥 Клиника-Анализатор")
    st.markdown("*Анализ удалённости пациентов от клиники*")
    st.markdown("---")

    # Боковая панель
    clinic_city, clinic_street, clinic_house = sidebar()

    # Основная зона
    st.header("📁 Загрузка отчёта")
    uploaded_file = st.file_uploader(
        "Загрузите Excel-файл (отчёт со списком адресов пациентов)",
        type=["xlsx", "xls"],
        help="Система автоматически найдёт лист с данными и определит колонки город/улица/дом"
    )

    if uploaded_file is not None:
        if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
            success = process_excel(uploaded_file, clinic_city, clinic_street, clinic_house)
            if success:
                st.balloons()
                cache_size = len(st.session_state['cache'])
                st.info(
                    f"📦 Обработка завершена. В кэше теперь **{cache_size}** адресов. "
                    "Не забудьте скачать обновлённый geo_cache.json внизу страницы!"
                )

    # Показываем результаты, если они есть
    if st.session_state['processing_done'] and st.session_state['agg'] is not None:
        # Показываем баннер, если в кэше появились новые адреса
        cache_size = len(st.session_state['cache'])
        st.success(
            f"🗺 Кэш геокодирования обновлён! Всего адресов в кэше: **{cache_size}**. "
            "Обязательно скачайте geo_cache.json в разделе '💾 Экспорт данных' — "
            "иначе при закрытии вкладки данные сгорят."
        )
        show_dashboard()
        export_data()

    # Футер
    st.markdown("---")
    st.caption("Клиника-Анализатор v1.0 | Streamlit Edition | Геокодеры: Photon + Nominatim")

if __name__ == "__main__":
    main()
