import requests
import streamlit as st
import pandas as pd
import numpy as np


# ==================================================
# 1. Page configuration
# ==================================================

st.set_page_config(
    page_title="Gold Pattern Explorer",
    layout="wide"
)

st.title("🟠 Gold Pattern Explorer")
st.write("Historical Gold Pattern Matching & Backtesting")
st.info(
    "يجلب التطبيق بيانات الذهب تلقائيًا عند الإمكان، مع الاحتفاظ بإمكانية "
    "رفع ملف CSV يدويًا كخيار بديل."
)

API_URL = "https://xaus.com/api/v1/chart"
MIN_HISTORY_MONTHS = 6
MAX_LATEST_CANDLE_AGE_HOURS = 72


# ==================================================
# 2. Data helpers
# ==================================================

def validate_history(dataframe):
    """Normalize OHLC history and reject invalid or insufficient data."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("لم تصل أي شموع صالحة من مصدر البيانات.")

    history = dataframe.copy()
    history.columns = [str(column).strip().lower() for column in history.columns]

    # Accept both CSV names and XAUS API's compact OHLCV names.
    history = history.rename(columns={
        "t": "timestamp",
        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })

    required_columns = ["timestamp", "open", "high", "low", "close"]
    missing_columns = [column for column in required_columns if column not in history.columns]
    if missing_columns:
        raise ValueError(
            "الأعمدة المطلوبة غير موجودة: " + ", ".join(missing_columns)
        )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"], utc=True, errors="coerce"
    )

    price_columns = ["open", "high", "low", "close"]
    for column in price_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    if "volume" in history.columns:
        history["volume"] = pd.to_numeric(history["volume"], errors="coerce")
    else:
        history["volume"] = np.nan

    history = history.dropna(subset=["timestamp"] + price_columns)
    history = history.sort_values("timestamp")
    history = history.drop_duplicates(subset=["timestamp"], keep="last")

    valid_ohlc = (
        (history["high"] >= history["low"])
        & (history["high"] >= history["open"])
        & (history["high"] >= history["close"])
        & (history["low"] <= history["open"])
        & (history["low"] <= history["close"])
        & (history[price_columns] > 0).all(axis=1)
    )
    history = history.loc[valid_ohlc].reset_index(drop=True)

    if history.empty:
        raise ValueError("لم تبقَ أي شموع صالحة بعد التحقق من الأسعار.")

    # Require at least six calendar months between the first and last candles.
    first_timestamp = history["timestamp"].min()
    last_timestamp = history["timestamp"].max()
    six_months_before_last = last_timestamp - pd.DateOffset(months=MIN_HISTORY_MONTHS)
    if first_timestamp > six_months_before_last:
        raise ValueError(
            "الفترة التاريخية المستلمة أقل من ستة أشهر. "
            f"الفترة المتاحة: {first_timestamp} — {last_timestamp}."
        )

    # The latest candle must not be excessively old. A 72-hour threshold allows
    # normal weekend/holiday gaps but rejects obviously stale automatic data.
    age = pd.Timestamp.now(tz="UTC") - last_timestamp
    if age > pd.Timedelta(hours=MAX_LATEST_CANDLE_AGE_HOURS):
        raise ValueError(
            "آخر شمعة قديمة جدًا للاستخدام التلقائي "
            f"(عمرها التقريبي {age.total_seconds() / 3600:.1f} ساعة)."
        )

    return history


def api_timestamp(value):
    """Parse XAUS timestamps expressed as Unix seconds/milliseconds or ISO text."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return pd.NaT
        unit = "ms" if abs(float(value)) >= 1e12 else "s"
        return pd.to_datetime(value, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def fetch_automatic_history():
    """Fetch two years of hourly XAU/USD candles and validate the response."""
    response = requests.get(
        API_URL,
        params={"symbol": "xau", "range": "2y", "interval": "1h"},
        timeout=25,
        headers={"User-Agent": "GoldPatternExplorer/1.0"},
    )
    response.raise_for_status()
    payload = response.json()

    data_state = payload.get("data_state", {}) if isinstance(payload, dict) else {}
    api_status = str(data_state.get("status", "")).lower() if isinstance(data_state, dict) else ""
    if api_status == "unavailable":
        raise ValueError("مصدر البيانات يعلن أن البيانات غير متاحة حاليًا.")
    if api_status == "stale":
        raise ValueError("مصدر البيانات يعلن أن البيانات التاريخية قديمة؛ لم تُستخدم تلقائيًا.")

    if isinstance(payload, list):
        points = payload
    elif isinstance(payload, dict):
        points = None
        for key in ("points", "data", "candles", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                points = value
                break
        if points is None:
            # Handle a standard chart-response wrapper if the upstream changes shape.
            chart = payload.get("chart")
            if isinstance(chart, dict):
                result = chart.get("result")
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    timestamps = result[0].get("timestamp", [])
                    quote = (result[0].get("indicators", {}).get("quote") or [{}])[0]
                    points = [
                        {
                            "t": timestamps[i],
                            "o": (quote.get("open") or [None] * len(timestamps))[i],
                            "h": (quote.get("high") or [None] * len(timestamps))[i],
                            "l": (quote.get("low") or [None] * len(timestamps))[i],
                            "c": (quote.get("close") or [None] * len(timestamps))[i],
                            "v": (quote.get("volume") or [None] * len(timestamps))[i],
                        }
                        for i in range(len(timestamps))
                    ]
        if points is None:
            raise ValueError("صيغة استجابة مصدر البيانات غير معروفة.")
    else:
        raise ValueError("استجابة مصدر البيانات ليست بصيغة JSON المتوقعة.")

    if not points:
        raise ValueError("مصدر البيانات لم يُرجع أي شموع.")

    raw = pd.DataFrame(points)
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    timestamp_column = next(
        (column for column in ("t", "timestamp", "time", "datetime", "date") if column in raw.columns),
        None,
    )
    if timestamp_column is None:
        raise ValueError("استجابة مصدر البيانات لا تحتوي على وقت الشمعة.")

    raw["timestamp"] = raw[timestamp_column].map(api_timestamp)
    raw = raw.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    if timestamp_column != "timestamp" and timestamp_column in raw.columns:
        # The parsed timestamp column above is authoritative.
        raw = raw.drop(columns=[timestamp_column])

    history = validate_history(raw)
    return history, data_state


def load_csv_history(uploaded_file):
    """Read an uploaded CSV using the same validation rules as automatic data."""
    csv_data = pd.read_csv(uploaded_file)
    return validate_history(csv_data)


# ==================================================
# 3. Automatic data fetching, with manual fallback
# ==================================================

if "auto_fetch_attempted" not in st.session_state:
    st.session_state["auto_fetch_attempted"] = True
    try:
        with st.spinner("جارٍ جلب بيانات الذهب التاريخية تلقائيًا..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
    except Exception as error:
        st.session_state["auto_history_data"] = None
        st.session_state["auto_data_state"] = {}
        st.session_state["auto_fetch_error"] = str(error)
        st.session_state["source_mode"] = "Upload CSV"

st.header("1. Data Source")

if st.button("🔄 Refresh automatic gold data"):
    try:
        with st.spinner("جارٍ تحديث بيانات الذهب التاريخية..."):
            auto_history, auto_state = fetch_automatic_history()
        st.session_state["auto_history_data"] = auto_history
        st.session_state["auto_data_state"] = auto_state
        st.session_state["auto_fetch_error"] = None
        st.session_state["source_mode"] = "Automatic data"
        st.success("تم تحديث البيانات التلقائية بنجاح.")
    except Exception as error:
        st.session_state["auto_fetch_error"] = str(error)
        st.warning(
            "تعذّر تحديث البيانات التلقائية. يمكنك متابعة العمل بملف CSV اليدوي. "
            f"التفاصيل: {error}"
        )

source_mode = st.radio(
    "Choose data source",
    options=["Automatic data", "Upload CSV"],
    horizontal=True,
    key="source_mode",
)

history = None

if source_mode == "Automatic data":
    auto_history = st.session_state.get("auto_history_data")
    if isinstance(auto_history, pd.DataFrame) and not auto_history.empty:
        history = auto_history.copy()
        st.success(
            f"تم تحميل {len(history):,} شمعة تاريخية تلقائيًا. "
            "المصدر: XAUS (XAU/USD، إطار الساعة)."
        )
        st.caption(
            "الفترة التاريخية (UTC): "
            f"{history['timestamp'].min()} — {history['timestamp'].max()}"
        )
        state = st.session_state.get("auto_data_state", {})
        if isinstance(state, dict) and state.get("as_of"):
            st.caption(f"حالة/وقت المصدر حسب API: {state.get('status', 'غير محدد')} — {state['as_of']}")
        st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
    else:
        error_message = st.session_state.get("auto_fetch_error")
        st.warning(
            "البيانات التلقائية غير متاحة حاليًا. اختر «Upload CSV» "
            "للعودة إلى الطريقة القديمة، أو اضغط زر التحديث لإعادة المحاولة."
        )
        if error_message:
            st.caption(f"تفاصيل فشل الجلب التلقائي: {error_message}")
else:
    history_file = st.file_uploader(
        "Upload historical gold data (CSV)",
        type=["csv"],
        key="history",
        help="يجب أن يحتوي الملف على timestamp, open, high, low, close."
    )
    if history_file is not None:
        try:
            history = load_csv_history(history_file)
            st.session_state["csv_history_data"] = history
            st.success(f"Loaded {len(history):,} valid candles from CSV.")
            st.caption(
                "الفترة التاريخية (UTC): "
                f"{history['timestamp'].min()} — {history['timestamp'].max()}"
            )
            st.dataframe(history.tail(5), use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"تعذرت قراءة ملف CSV أو التحقق منه: {error}")
    elif isinstance(st.session_state.get("csv_history_data"), pd.DataFrame):
        history = st.session_state["csv_history_data"].copy()
        st.info("تم الاحتفاظ بملف CSV الذي رفعته سابقًا خلال هذه الجلسة.")

if isinstance(history, pd.DataFrame):
    st.session_state["history_data"] = history.copy()
else:
    st.session_state.pop("history_data", None)


# ==================================================
# 4. Historical pattern matching (existing method retained)
# ==================================================

st.divider()
st.header("2. Historical Matching")

if "history_data" not in st.session_state:
    st.info("اختر مصدر البيانات وحمّل البيانات قبل بدء البحث عن الأنماط.")
else:
    history = st.session_state["history_data"].copy()

    pattern_length = st.selectbox(
        "Pattern length (hourly candles)",
        [48, 72, 96, 144],
        index=3,
        key="matching_pattern_length"
    )

    # Changing the source or pattern length should not reuse an old target
    # index that would move the target away from the latest candles.
    active_source = st.session_state.get("_active_data_source")
    previous_pattern_length = st.session_state.get("_previous_pattern_length")
    if active_source != source_mode or (
        previous_pattern_length is not None
        and previous_pattern_length != pattern_length
    ):
        for length in [48, 72, 96, 144]:
            st.session_state.pop(f"matching_target_start_{length}", None)
    st.session_state["_active_data_source"] = source_mode
    st.session_state["_previous_pattern_length"] = pattern_length

    minimum_candles = pattern_length * 2 + 1

    if len(history) < minimum_candles:
        st.warning(
            f"At least {minimum_candles} valid candles are needed "
            f"for a {pattern_length}-candle pattern and an earlier historical comparison."
        )
    else:
        max_start = len(history) - pattern_length
        target_start = st.number_input(
            "Target starting candle index",
            min_value=0,
            max_value=max_start,
            value=max_start,
            step=1,
            key=f"matching_target_start_{pattern_length}"
        )

        target_start = int(target_start)
        target_end = target_start + pattern_length
        target = history.iloc[target_start:target_end].copy()

        st.subheader("Selected target")
        st.write(
            "Target period (UTC):",
            str(target["timestamp"].iloc[0]),
            "—",
            str(target["timestamp"].iloc[-1])
        )
        st.caption(
            "عند اختيار آخر فهرس، تكون العينة هي أحدث الشموع المتاحة "
            "بعدد يساوي طول النمط المختار."
        )
        st.line_chart(target.set_index("timestamp")["close"])

        if len(target) != pattern_length:
            st.warning("The selected target does not contain the required number of candles.")
        else:
            target_close = target["close"].to_numpy(dtype=float)
            target_base = target_close[0]

            if not np.isfinite(target_close).all() or target_base == 0:
                st.error("The target pattern contains invalid prices.")
            else:
                target_shape = (target_close / target_base) - 1
                results = []

                # Historical candidates end before the selected target begins.
                for start in range(0, target_start - pattern_length + 1):
                    end = start + pattern_length
                    candidate = history.iloc[start:end]
                    if len(candidate) != pattern_length:
                        continue

                    candidate_close = candidate["close"].to_numpy(dtype=float)
                    if not np.isfinite(candidate_close).all():
                        continue
                    candidate_base = candidate_close[0]
                    if candidate_base == 0:
                        continue

                    candidate_shape = (candidate_close / candidate_base) - 1
                    rmse = float(np.sqrt(np.mean((target_shape - candidate_shape) ** 2)))
                    similarity = 100 / (1 + rmse * 100)
                    future = history.iloc[end:min(end + 12, len(history))].copy()

                    results.append({
                        "start": start,
                        "end": end,
                        "rmse": rmse,
                        "similarity": similarity,
                        "future": future,
                    })

                results.sort(key=lambda item: item["rmse"])
                st.subheader("Historical matching results")

                if not results:
                    st.warning("No eligible historical matches were found. Try another target starting index.")
                else:
                    st.success(f"Found {len(results):,} historical matches.")
                    top_n = st.slider(
                        "Number of matches to display",
                        min_value=1,
                        max_value=min(20, len(results)),
                        value=min(10, len(results))
                    )

                    summary = []
                    for rank, result in enumerate(results[:top_n], start=1):
                        matched = history.iloc[result["start"]:result["end"]]
                        summary.append({
                            "Rank": rank,
                            "Similarity (%)": round(result["similarity"], 2),
                            "Start (UTC)": str(matched["timestamp"].iloc[0]),
                            "End (UTC)": str(matched["timestamp"].iloc[-1]),
                        })

                    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
                    st.caption(
                        "Similarity is a mathematical shape score, not a probability or a price forecast."
                    )

                    selected_rank = st.selectbox(
                        "Inspect a historical match",
                        options=list(range(1, top_n + 1)),
                        format_func=lambda rank: f"Match {rank}"
                    )
                    selected = results[selected_rank - 1]
                    st.write("Similarity score:", f"{selected['similarity']:.2f}%")

                    matched = history.iloc[selected["start"]:selected["end"]].copy()
                    st.subheader("Matched historical pattern")
                    st.line_chart(matched.set_index("timestamp")["close"])

                    future = selected["future"]
                    if not future.empty:
                        st.subheader("What happened afterward")
                        st.line_chart(future.set_index("timestamp")["close"])
                        st.dataframe(
                            future[["timestamp", "open", "high", "low", "close"]],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No subsequent candles are available after this historical match.")


# ==================================================
# 5. Backtesting placeholder (unchanged in scope)
# ==================================================

st.divider()
st.header("3. Backtesting")
st.info(
    "A proper walk-forward backtest with performance statistics has not yet been implemented. "
    "Historical similarity scores alone do not establish whether this method is profitable."
    )
                          
