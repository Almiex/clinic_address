# -*- coding: utf-8 -*-
"""
🏥 КЛИНИКА-АНАЛИЗАТОР — Streamlit Cloud Edition v1.1
Анализ удалённости пациентов от клиники с геокодированием,
сегментацией и фильтром по датам.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Photon, Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from scipy.ndimage import gaussian_filter1d
import time
import json
import os
import re
import io
from datetime import datetime, date

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
COLORS_MAP = {
    "0–2 км": "#2ecc71",
    "2–5 км": "#3498db",
    "5–7 км": "#f1c40f",
    "7–10 км": "#e67e22",
    "10+ км": "#e74c3c",
    "Другие города": "#9b59b6",
    "Нет данных": "#95a5a6",
}

# ═══════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════════════

def init_session_state():
    """Инициализация session_state для хранения данных между ререндерами."""
    defaults = {
        'cache': {},
        'df_processed': None,
        'clinic_coord': None,
        'clinic_addr': None,
        'clinic_norm_city': None,
        'date_range_str': '',
        'processing_done': False,
        'col_map': {},
        'min_date': None,
        'max_date': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def is_missing(value, is_house=False):
    """
    Проверяет, является ли значение пустым/некорректным.
    Для дома дополнительно проверяет наличие хотя бы одной буквы/цифры.
    """
    if pd.isna(value):
        return True
    s = str(value).strip().lower()
    if s in ('', 'nan', 'none', 'null', '-', '—', '–', 'нет', 'отсутствует',
             'б/н', 'бн', 'б.н.', 'б.н', '0', 'нет данных', 'не указан',
             'без номера', 'б/д', 'бд', 'б.д.', 'б.д', 'н/д', 'нд'):
        return True
    if is_house:
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', s):
            return True
        if not re.search(r'[a-zа-яё0-9]', s):
            return True
    return False

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

def find_date_col(cols):
    """
    Ищет колонку с датой обращения/визита/оплаты.
    Явно исключает колонки с датой рождения и другими ложными совпадениями.
    """
    exclude = ['рожд', 'birth', 'д.р.', 'др ', 'д/р', 'дата рожд', 'date of birth', 'birthdate']
    keys = [
        'дата обращ', 'дата оплат', 'дата приёма', 'дата приема', 'дата визита',
        'дата посещ', 'дата консул', 'дата процед', 'дата операц',
        'дата заезда', 'дата выписк', 'дата планир', 'дата назнач',
        'дата регистр', 'дата записи', 'дата событ', 'дата акт',
        'date of visit', 'visit date', 'appointment date', 'service date',
        'payment date', 'receipt date', 'admission date', 'discharge date'
    ]
    for c in cols:
        c_lower = str(c).lower()
        # Сначала проверяем исключения
        if any(ex in c_lower for ex in exclude):
            continue
        # Теперь ищем ключи даты обращения
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

def compute_agg(df, col_map):
    """Пересчёт агрегации по сегментам для произвольного датафрейма."""
    col_sum = col_map.get('sum')
    col_serv = col_map.get('serv')

    if col_sum and col_sum in df.columns:
        df[col_sum] = pd.to_numeric(df[col_sum], errors='coerce').fillna(0)
    if col_serv and col_serv in df.columns:
        df[col_serv] = pd.to_numeric(df[col_serv], errors='coerce').fillna(0)

    agg = df.groupby('segment').agg(
        patients=('segment', 'count'),
        services=(col_serv, 'sum') if (col_serv and col_serv in df.columns) else ('segment', 'count'),
        revenue=(col_sum, 'sum') if (col_sum and col_sum in df.columns) else ('distance_m', 'sum'),
        avg_dist=('distance_m', 'mean')
    ).reindex(SEGMENT_ORDER).fillna(0)

    total_patients = agg['patients'].sum()
    total_revenue = agg['revenue'].sum()
    agg['patients_pct'] = (agg['patients'] / total_patients * 100).round(1) if total_patients > 0 else 0
    agg['revenue_pct'] = (agg['revenue'] / total_revenue * 100).round(1) if total_revenue > 0 else 0
    return agg

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

def _cell_to_str(x):
    """Конвертирует ячейку Excel в строку, datetime → dd.mm.yyyy."""
    if pd.isna(x):
        return ""
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.strftime('%d.%m.%Y')
    return str(x)

def _extract_date_from_cell(cell_text):
    """Ищет дату dd.mm.yyyy в тексте ячейки. Возвращает date или None."""
    if not cell_text:
        return None
    s = cell_text.strip()
    # Паттерны: 01.01.2026, 01-01-2026, 01/01/2026
    m = re.search(r'(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})', s)
    if m:
        d, mth, y = m.group(1), m.group(2), m.group(3)
        try:
            return datetime(int(y), int(mth), int(d)).date()
        except ValueError:
            pass
    return None

def extract_date_range(df_raw):
    """Ищем даты периода в первых 10 строках, первых 5 столбцах (заголовок отчёта).

    Проверяем каждую ячейку отдельно — даты обычно в R3C1 (С:) и R3C2 (ПО:).
    """
    date_start = None
    date_end = None

    for i in range(min(10, len(df_raw))):
        row = df_raw.iloc[i]
        for j in range(min(5, len(row))):
            cell = _cell_to_str(row.iloc[j])
            if not cell:
                continue
            cell_lower = cell.lower()
            d = _extract_date_from_cell(cell)
            if d is None:
                continue
            # С: / С / начало / от
            if any(k in cell_lower for k in ['с:', 'с ', 'начало', 'от ', 'с	']):
                date_start = d
            # ПО: / по / конец / до
            elif any(k in cell_lower for k in ['по:', 'по ', 'конец', 'до ']):
                date_end = d

    if date_start and date_end:
        s_str = date_start.strftime('%d.%m.%Y')
        e_str = date_end.strftime('%d.%m.%Y')
        return f"Период анализа: {s_str} – {e_str}", date_start, date_end
    return "", None, None

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

    clinic_addr = build_address(clinic_house, clinic_street, clinic_city)
    clinic_norm_city = normalize_city(clinic_city)
    st.session_state['clinic_addr'] = clinic_addr
    st.session_state['clinic_norm_city'] = clinic_norm_city

    if not clinic_city or not clinic_street or not clinic_house:
        st.error("❌ Заполните все поля адреса клиники в боковой панели!")
        return False

    st.info(f"🔍 Адрес клиники: {clinic_addr}")

    xl = pd.ExcelFile(uploaded_file)
    sheet = 'Лист 2' if 'Лист 2' in xl.sheet_names else (xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0])
    st.write(f"📄 Работаем с листом: **{sheet}**")

    df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)

    date_range_str, report_start, report_end = extract_date_range(df_raw)
    st.session_state['date_range_str'] = date_range_str
    st.session_state['report_date_start'] = report_start
    st.session_state['report_date_end'] = report_end
    if date_range_str:
        st.success(f"📅 {date_range_str}")

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

    col_city = find_col(df.columns, ['город'])
    col_street = find_col(df.columns, ['улица'])
    col_house = find_col(df.columns, ['дом'])
    col_serv = find_col(df.columns, ['усл', 'услуг'])
    col_sum = find_col(df.columns, ['сумма', 'оплат'])
    col_date = find_date_col(df.columns)

    st.session_state['col_map'] = {
        'city': col_city, 'street': col_street, 'house': col_house,
        'serv': col_serv, 'sum': col_sum, 'date': col_date
    }

    if not col_city or not col_street or not col_house:
        st.error("❌ Не удалось найти обязательные колонки (город / улица / дом).")
        st.write(f"Найдены колонки: {list(df.columns)}")
        return False

    st.write(f"🔎 Найдены колонки: город=`{col_city}`, улица=`{col_street}`, дом=`{col_house}`")
    if col_date:
        st.write(f"📅 Найдена колонка дат обращений: `{col_date}`")
    else:
        st.info("ℹ️ Колонка с датами обращений/визитов не найдена — фильтр по датам будет недоступен.")

    df[col_city] = df[col_city].astype(str).replace('nan', '').replace('None', '')
    df[col_street] = df[col_street].astype(str).replace('nan', '').replace('None', '')
    df[col_house] = df[col_house].astype(str).replace('nan', '').replace('None', '')

    # --- Фильтр неполных / некорректных адресов ---
    bad_mask = (
        df[col_city].apply(lambda x: is_missing(x)) |
        df[col_street].apply(lambda x: is_missing(x)) |
        df[col_house].apply(lambda x: is_missing(x, is_house=True))
    )

    df['is_valid'] = ~bad_mask
    df.loc[bad_mask, 'segment'] = 'Нет данных'

    excluded = df[bad_mask].copy()
    df_good = df[~bad_mask].copy()

    df_good['norm_city'] = df_good[col_city].apply(normalize_city)

    missing_city = df[col_city].apply(lambda x: is_missing(x)).sum()
    missing_street = df[col_street].apply(lambda x: is_missing(x)).sum()
    missing_house = df[col_house].apply(lambda x: is_missing(x, is_house=True)).sum()

    st.write(f"🚫 Исключено → 'Нет данных': **{len(excluded)}** чел.")
    with st.expander("📋 Детализация причин исключения"):
        st.write(f"   • Нет города: {missing_city}")
        st.write(f"   • Нет улицы: {missing_street}")
        st.write(f"   • Нет/некорректный дом: {missing_house}")
        st.caption("Примечание: одна запись может попадать сразу в несколько категорий.")
    st.write(f"✅ Будет обработано: **{len(df_good)}** чел.")

    # --- Геокодирование ---
    cache = st.session_state['cache']
    geo_photon = Photon(user_agent="clinic_analyzer_streamlit", timeout=15)
    geo_nominatim = Nominatim(user_agent="clinic_analyzer_streamlit", timeout=20)

    with st.spinner("🔍 Геокодирование клиники..."):
        if clinic_addr not in cache or cache[clinic_addr] is None:
            geocode_with_fallback(geo_photon, geo_nominatim, clinic_addr, cache)
        clinic_coord = cache.get(clinic_addr)

    if not clinic_coord:
        st.error("❌ Не удалось найти координаты клиники. Проверьте адрес.")
        return False

    st.session_state['clinic_coord'] = clinic_coord
    st.success(f"✅ Координаты клиники: {clinic_coord}")

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

    df = pd.concat([df_good, excluded], ignore_index=True)
    df['segment'] = df['segment'].fillna('Нет данных')

    df_ok = df.dropna(subset=['distance_m']).copy()
    st.write(f"📊 Успешно геокодировано: **{len(df_ok)}** из **{len(df_good)}** валидных")
    st.write(f"📊 Всего в отчёте: **{len(df)}** (включая **{len(excluded)}** без адреса)")

    # --- Даты для фильтра берутся из заголовка отчёта, не из колонок данных ---
    # (чтобы не путать дату рождения с датой визита)
    if col_date and col_date in df.columns:
        df[col_date] = pd.to_datetime(df[col_date], errors='coerce', dayfirst=True)
        st.write(f"📅 Найдена колонка дат визита/обращения: `{col_date}`")
    else:
        st.info("ℹ️ Колонка с датами визита/обращения не найдена — фильтр по датам будет декларативным.")

    st.session_state['df_processed'] = df
    st.session_state['processing_done'] = True

    return True


# ═══════════════════════════════════════════════════════════════════════
#  КАРТА (Folium)
# ═══════════════════════════════════════════════════════════════════════

def build_map(df, clinic_coord, clinic_addr):
    """Строит интерактивную карту с клиникой и точками пациентов."""

    # Фильтруем только записи с валидными координатами
    df_map = df[df['coords'].apply(lambda c: isinstance(c, (list, tuple)) and len(c) == 2)].copy()

    if len(df_map) == 0:
        return None

    df_map['lat'] = df_map['coords'].apply(lambda c: c[0])
    df_map['lon'] = df_map['coords'].apply(lambda c: c[1])

    center = clinic_coord
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    # Клиника
    folium.Marker(
        location=center,
        popup="<b>🏥 Клиника</b><br>" + str(clinic_addr),
        tooltip="Клиника",
        icon=folium.Icon(color="red", icon="plus-sign", prefix="glyphicon")
    ).add_to(m)

    # Зоны вокруг клиники
    for radius, color, label in [(2000, '#2ecc71', '2 км'), (5000, '#3498db', '5 км'),
                                  (7000, '#f1c40f', '7 км'), (10000, '#e67e22', '10 км')]:
        folium.Circle(
            location=center,
            radius=radius,
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.08,
            popup=label + " от клиники"
        ).add_to(m)

    # Кластеризация точек пациентов
    marker_cluster = MarkerCluster(name="Пациенты").add_to(m)

    for _, row in df_map.iterrows():
        seg = row.get('segment', 'Нет данных')
        color = COLORS_MAP.get(seg, '#95a5a6')
        dist_km = row.get('distance_km', '—')
        addr = row.get('geo_address', '—')

        popup_html = (
            "<b>Сегмент:</b> " + str(seg) + "<br>"
            "<b>Расстояние:</b> " + str(dist_km) + " км<br>"
            "<b>Адрес:</b> " + str(addr)
        )

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=str(seg) + " | " + str(dist_km) + " км"
        ).add_to(marker_cluster)

    # Легенда
    legend_html = (
        '<div style="position: fixed; bottom: 50px; left: 50px; width: 160px; '
        'background-color: white; border:2px solid grey; z-index:9999; '
        'font-size:12px; padding: 10px; border-radius: 6px; '
        'box-shadow: 2px 2px 5px rgba(0,0,0,0.2);">'
        '<b>Легенда</b><br>'
        '<i style="color:#e74c3c;">●</i> Клиника<br>'
        '<i style="color:#2ecc71;">●</i> 0–2 км<br>'
        '<i style="color:#3498db;">●</i> 2–5 км<br>'
        '<i style="color:#f1c40f;">●</i> 5–7 км<br>'
        '<i style="color:#e67e22;">●</i> 7–10 км<br>'
        '<i style="color:#e74c3c;">●</i> 10+ км<br>'
        '<i style="color:#9b59b6;">●</i> Другие города<br>'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

# ═══════════════════════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ (Plotly)
# ═══════════════════════════════════════════════════════════════════════

def show_dashboard(df, agg, date_start=None, date_end=None):
    clinic_addr = st.session_state['clinic_addr']
    date_range_str = st.session_state['date_range_str']

    title_text = f'Анализ клиентов относительно клиники<br>{clinic_addr}'
    if date_range_str:
        title_text += f'<br>{date_range_str}'
    if date_start and date_end:
        title_text += f'<br>Фильтр: {date_start} – {date_end}'

    st.markdown("---")
    st.header("📊 Результаты анализа")

    # Сводная таблица
    st.subheader("Сводка по сегментам")
    agg_display = agg.copy()
    agg_display['avg_dist'] = agg_display['avg_dist'].apply(lambda x: f"{x/1000:.1f} км" if x > 0 else "—")
    agg_display['revenue'] = agg_display['revenue'].apply(lambda x: f"{int(x):,} ₽".replace(",", " "))
    agg_display['patients'] = agg_display['patients'].astype(int)
    st.dataframe(agg_display, use_container_width=True)

    # --- Plotly: Количество пациентов ---
    st.subheader("Графики")

    nonzero = agg[agg['patients'] > 0].copy()
    nonzero['color'] = nonzero.index.map(lambda s: COLORS_MAP.get(s, '#95a5a6'))

    fig1 = px.bar(
        nonzero.reset_index(),
        x='segment',
        y='patients',
        color='segment',
        color_discrete_map=COLORS_MAP,
        text=nonzero.apply(lambda r: f"{int(r['patients'])}<br>({r['patients_pct']}%)", axis=1),
        title='Количество пациентов',
        labels={'patients': 'Человек', 'segment': ''}
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig1, use_container_width=True)

    # --- Plotly: Сумма оплат ---
    nonzero_rev = agg[agg['revenue'] > 0].copy()
    if len(nonzero_rev) > 0:
        fig2 = px.bar(
            nonzero_rev.reset_index(),
            x='segment',
            y='revenue',
            color='segment',
            color_discrete_map=COLORS_MAP,
            text=nonzero_rev.apply(lambda r: f"{int(r['revenue']):,} ₽<br>({r['revenue_pct']}%)", axis=1),
            title='Сумма оплат',
            labels={'revenue': 'Руб.', 'segment': ''}
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig2, use_container_width=True)

    # --- Plotly: Круговые диаграммы (2 рядом) ---
    col_pie1, col_pie2 = st.columns(2)

    with col_pie1:
        pie_pat = agg[agg['patients'] > 0]
        if len(pie_pat) > 0:
            fig3 = px.pie(
                pie_pat.reset_index(),
                names='segment',
                values='patients',
                color='segment',
                color_discrete_map=COLORS_MAP,
                title='Доля пациентов',
                hole=0.4
            )
            fig3.update_traces(textinfo='percent+label', pull=[0.02]*len(pie_pat))
            fig3.update_layout(height=450)
            st.plotly_chart(fig3, use_container_width=True)

    with col_pie2:
        pie_rev = agg[agg['revenue'] > 0]
        if len(pie_rev) > 0:
            fig4 = px.pie(
                pie_rev.reset_index(),
                names='segment',
                values='revenue',
                color='segment',
                color_discrete_map=COLORS_MAP,
                title='Доля выручки',
                hole=0.4
            )
            fig4.update_traces(textinfo='percent+label', pull=[0.02]*len(pie_rev))
            fig4.update_layout(height=450)
            st.plotly_chart(fig4, use_container_width=True)

    # --- Плавный график распределения (Plotly) ---
    st.subheader("📈 Распределение по удалённости (только основной город)")

    df_local = df[df['segment'] != 'Другие города'].copy()
    df_local = df_local[df_local['segment'] != 'Нет данных'].copy()
    distances = df_local['distance_km'].dropna()
    distances_plot = distances[distances <= 50].copy()

    if len(distances_plot) > 0:
        x_max = distances_plot.quantile(0.995)
        if x_max < 5:
            x_max = 20

        bins = np.arange(0, x_max + 0.1, 0.1)
        counts, edges = np.histogram(distances_plot, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        counts_smooth = gaussian_filter1d(counts.astype(float), sigma=15)

        city_name = st.session_state['clinic_norm_city'].title()
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=centers, y=counts_smooth,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#2980b9', width=2.5),
            fillcolor='rgba(52, 152, 219, 0.25)',
            name='Пациенты'
        ))
        fig5.update_layout(
            title=f'Распределение по удалённости (только {city_name})',
            xaxis_title='Расстояние от клиники, км',
            yaxis_title='Количество пациентов',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig5, use_container_width=True)

        outliers = distances[distances > x_max]
        if len(outliers) > 0:
            st.caption(f"Ось X ограничена {x_max:.1f} км. Выбросов за пределами: {len(outliers)} чел.")
    else:
        st.info("Нет данных для построения графика распределения")

    # --- Карта ---
    st.markdown("---")
    st.subheader("🗺 Карта пациентов")

    clinic_coord = st.session_state.get('clinic_coord')
    clinic_addr = st.session_state.get('clinic_addr')

    if clinic_coord and clinic_addr:
        with st.spinner("🗺 Построение карты..."):
            m = build_map(df, clinic_coord, clinic_addr)
        if m:
            st.write(f"📍 Точек на карте: **{len(df[df['coords'].apply(lambda c: isinstance(c, (list, tuple)) and len(c) == 2)])}**")
            st_folium(m, width=1200, height=700, key="patient_map")
            st.caption("💡 Нажмите на кластер, чтобы раскрыть точки. Цвета соответствуют сегментам удалённости.")
        else:
            st.warning("⚠️ Нет координат для построения карты.")
    else:
        st.warning("⚠️ Координаты клиники не найдены.")

# ═══════════════════════════════════════════════════════════════════════
#  ЭКСПОРТ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════

def export_data(df, agg):
    st.markdown("---")
    st.header("💾 Экспорт данных")

    cache = st.session_state['cache']

    col1, col2, col3 = st.columns(3)

    with col1:
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
            if col_map.get('sum'):
                display_cols.append(col_map['sum'])
        elif selected_segment == 'Другие города':
            display_cols = [col_map['city'], col_map['street'], col_map['house'], 'distance_km']
            if col_map.get('sum'):
                display_cols.append(col_map['sum'])
        else:
            display_cols = ['geo_address', 'distance_km']
            if col_map.get('sum'):
                display_cols.append(col_map['sum'])

        display_cols = [c for c in display_cols if c in cohort.columns]
        st.dataframe(cohort[display_cols].head(50), use_container_width=True)

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

    clinic_city, clinic_street, clinic_house = sidebar()

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

    # Показываем результаты
    if st.session_state['processing_done'] and st.session_state['df_processed'] is not None:
        df_full = st.session_state['df_processed']
        col_map = st.session_state['col_map']
        col_date = col_map.get('date')

        # --- Фильтр по датам ---
        date_start = None
        date_end = None

        # ═══════════════════════════════════════════════════════════
        #  ФИЛЬТР ПО ДАТАМ — ЗАКОММЕНТИРОВАН (v1.2)
        #  Требует колонки с датами визита/обращения в данных.
        #  В текущем отчёте даты только в заголовке.
        # ═══════════════════════════════════════════════════════════
        # report_start = st.session_state.get('report_date_start')
        # report_end = st.session_state.get('report_date_end')
        # 
        # df = df_full.copy()
        # filter_active = False
        # date_start = None
        # date_end = None
        # 
        # if report_start and report_end:
        #     st.markdown("---")
        #     st.header("📅 Фильтр по датам")
        #     st.caption(f"Диапазон отчёта: {report_start.strftime('%d.%m.%Y')} – {report_end.strftime('%d.%m.%Y')}")
        #     
        #     c1, c2, c3 = st.columns([2, 2, 1])
        #     with c1:
        #         date_start = st.date_input(
        #             "Дата начала",
        #             value=report_start,
        #             min_value=report_start,
        #             max_value=report_end,
        #             key="filter_start_input"
        #         )
        #     with c2:
        #         date_end = st.date_input(
        #             "Дата окончания",
        #             value=report_end,
        #             min_value=report_start,
        #             max_value=report_end,
        #             key="filter_end_input"
        #         )
        #     with c3:
        #         st.write("")
        #         st.write("")
        #         apply_clicked = st.button("🔍 Применить фильтр", use_container_width=True, key="apply_filter_btn")
        #     
        #     if apply_clicked:
        #         if date_start > date_end:
        #             st.error("❌ Дата начала не может быть позже даты окончания!")
        #         else:
        #             col_date = col_map.get('date')
        #             if col_date and col_date in df_full.columns:
        #                 mask = (df_full[col_date].dt.date >= date_start) & (df_full[col_date].dt.date <= date_end)
        #                 st.session_state['filtered_df'] = df_full[mask].copy()
        #                 st.session_state['filter_applied'] = True
        #                 st.session_state['filter_start'] = date_start
        #                 st.session_state['filter_end'] = date_end
        #                 st.success(f"📊 Фильтр применён: **{len(st.session_state['filtered_df'])}** записей из **{len(df_full)}**")
        #             else:
        #                 st.session_state['filter_applied'] = False
        #                 st.info("ℹ️ Колонка с датами визита не найдена — фильтр применён декларативно.")
        #             st.rerun()
        #     
        #     if st.session_state.get('filter_applied') and st.session_state.get('filtered_df') is not None:
        #         df = st.session_state['filtered_df']
        #         date_start = st.session_state.get('filter_start')
        #         date_end = st.session_state.get('filter_end')
        #         filter_active = True
        #         
        #         c1, c2 = st.columns([3, 1])
        #         with c1:
        #             st.info(f"🔍 Активен фильтр: **{date_start} – {date_end}** | **{len(df)}** записей из **{len(df_full)}**")
        #         with c2:
        #             if st.button("❌ Сбросить фильтр", use_container_width=True, key="reset_filter_btn"):
        #                 st.session_state['filter_applied'] = False
        #                 st.session_state['filtered_df'] = None
        #                 st.session_state['filter_start'] = None
        #                 st.session_state['filter_end'] = None
        #                 st.rerun()
        # else:
        #     st.info("ℹ️ Даты периода не найдены в заголовке отчёта — фильтр по датам недоступен.")
        # ═══════════════════════════════════════════════════════════

        df = df_full.copy()
        filter_active = False
        date_start = None
        date_end = None

        # Пересчитываем агрегацию на лету
        agg = compute_agg(df, col_map)

        # Показываем дашборд
        show_dashboard(df, agg, date_start=date_start, date_end=date_end)
        export_data(df, agg)

        # Баннер о кэше
        cache_size = len(st.session_state['cache'])
        st.success(
            f"🗺 Кэш геокодирования обновлён! Всего адресов в кэше: **{cache_size}**. "
            "Обязательно скачайте geo_cache.json в разделе '💾 Экспорт данных' — "
            "иначе при закрытии вкладки данные сгорят."
        )

    st.markdown("---")
    st.caption("Клиника-Анализатор v1.1 | Streamlit Cloud | Plotly + GeoPy")

if __name__ == "__main__":
    main()
