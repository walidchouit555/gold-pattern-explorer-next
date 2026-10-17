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
    "ارفع بيانات الذهب التاريخية بصيغة CSV، "
    "ثم ابحث عن الأنماط التاريخية الأكثر تشابهًا."
)


# ==================================================
# 2. Historical data upload
# ==================================================

st.header("1. Historical Data")

history_file = st.file_uploader(
    "Upload historical gold data (CSV)",
    type=["csv"],
    key="history"
)

if history_file is not None:
    try:
        history = pd.read_csv(history_file)

        history.columns = [
            str(column).strip().lower()
            for column in history.columns
        ]

        required_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in history.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

            st.stop()

        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="coerce"
        )

        price_columns = [
            "open",
            "high",
            "low",
            "close"
        ]

        for column in price_columns:
            history[column] = pd.to_numeric(
                history[column],
                errors="coerce"
            )

        history = history.dropna(
            subset=["timestamp"] + price_columns
        )

        history = history.sort_values(
            "timestamp"
        )

        history = history.drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )

        history = history.reset_index(
            drop=True
        )

        # Reject invalid OHLC rows.
        valid_ohlc = (
            (history["high"] >= history["low"])
            & (history["high"] >= history["open"])
            & (history["high"] >= history["close"])
            & (history["low"] <= history["open"])
            & (history["low"] <= history["close"])
        )

        history = history.loc[
            valid_ohlc
        ].reset_index(drop=True)

        if history.empty:
            st.error(
                "No valid price candles were found in the CSV."
            )
            st.stop()

        st.session_state["history_data"] = history

        st.success(
            f"Loaded {len(history)} valid candles."
        )

        st.write(
            "Historical range (UTC):",
            history["timestamp"].min(),
            "—",
            history["timestamp"].max()
        )

        st.dataframe(
            history.tail(10),
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:
        st.error(
            f"Could not read CSV: {error}"
        )


# ==================================================
# 3. Historical pattern matching
# ==================================================

st.divider()
st.header("2. Historical Matching")

if "history_data" not in st.session_state:
    st.info(
        "Upload historical CSV data above to begin."
    )

else:
    history = st.session_state[
        "history_data"
    ].copy()

    pattern_length = st.selectbox(
        "Pattern length (hourly candles)",
        [48, 72, 96, 144],
        index=3,
        key="matching_pattern_length"
    )

    minimum_candles = pattern_length * 2 + 1

    if len(history) < minimum_candles:
        st.warning(
            f"At least {minimum_candles} valid candles are needed "
            f"for a {pattern_length}-candle pattern "
            "and an earlier historical comparison."
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

        target = history.iloc[
            target_start:target_end
        ].copy()

        st.subheader("Selected target")

        st.write(
            "Target period (UTC):",
            str(target["timestamp"].iloc[0]),
            "—",
            str(target["timestamp"].iloc[-1])
        )

        target_chart = target.set_index(
            "timestamp"
        )["close"]

        st.line_chart(target_chart)

        if len(target) != pattern_length:
            st.warning(
                "The selected target does not contain "
                "the required number of candles."
            )

        else:
            target_close = target[
                "close"
            ].to_numpy(dtype=float)

            target_base = target_close[0]

            if (
                not np.isfinite(target_close).all()
                or target_base == 0
            ):
                st.error(
                    "The target pattern contains invalid prices."
                )

            else:
                # Normalize the target so patterns are compared
                # by relative shape rather than absolute price.
                target_shape = (
                    target_close / target_base
                ) - 1

                results = []

                # Search only historical windows ending before
                # the selected target begins.
                for start in range(
                    0,
                    target_start - pattern_length + 1
                ):
                    end = start + pattern_length

                    candidate = history.iloc[
                        start:end
                    ]

                    if len(candidate) != pattern_length:
                        continue

                    candidate_close = candidate[
                        "close"
                    ].to_numpy(dtype=float)

                    if not np.isfinite(
                        candidate_close
                    ).all():
                        continue

                    candidate_base = candidate_close[0]

                    if candidate_base == 0:
                        continue

                    # Compare the normalized price shapes.
                    candidate_shape = (
                        candidate_close / candidate_base
                    ) - 1

                    rmse = float(
                        np.sqrt(
                            np.mean(
                                (
                                    target_shape
                                    - candidate_shape
                                ) ** 2
                            )
                        )
                    )

                    similarity = (
                        100 / (1 + rmse * 100)
                    )

                    # Store up to 12 candles after each match.
                    future = history.iloc[
                        end:min(end + 12, len(history))
                    ].copy()

                    results.append(
                        {
                            "start": start,
                            "end": end,
                            "rmse": rmse,
                            "similarity": similarity,
                            "future": future
                        }
                    )

                results.sort(
                    key=lambda item: item["rmse"]
                )

                st.subheader(
                    "Historical matching results"
                )

                if not results:
                    st.warning(
                        "No eligible historical matches were found. "
                        "Try another target starting index."
                    )

                else:
                    st.success(
                        f"Found {len(results)} historical matches."
                    )

                    top_n = st.slider(
                        "Number of matches to display",
                        min_value=1,
                        max_value=min(20, len(results)),
                        value=min(10, len(results))
                    )

                    summary = []

                    for rank, result in enumerate(
                        results[:top_n],
                        start=1
                    ):
                        matched = history.iloc[
                            result["start"]:result["end"]
                        ]

                        summary.append(
                            {
                                "Rank": rank,
                                "Similarity (%)": round(
                                    result["similarity"],
                                    2
                                ),
                                "Start (UTC)": str(
                                    matched["timestamp"].iloc[0]
                                ),
                                "End (UTC)": str(
                                    matched["timestamp"].iloc[-1]
                                )
                            }
                        )

                    st.dataframe(
                        pd.DataFrame(summary),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Similarity is a mathematical shape score. "
                        "It is not a probability or a price forecast."
                    )

                    selected_rank = st.selectbox(
                        "Inspect a historical match",
                        options=list(
                            range(1, top_n + 1)
                        ),
                        format_func=lambda rank: (
                            f"Match {rank}"
                        )
                    )

                    selected = results[
                        selected_rank - 1
                    ]

                    st.write(
                        "Similarity score:",
                        f"{selected['similarity']:.2f}%"
                    )

                    matched = history.iloc[
                        selected["start"]:selected["end"]
                    ].copy()

                    st.subheader(
                        "Matched historical pattern"
                    )

                    matched_chart = matched.set_index(
                        "timestamp"
                    )["close"]

                    st.line_chart(
                        matched_chart
                    )

                    future = selected["future"]

                    if not future.empty:
                        st.subheader(
                            "What happened afterward"
                        )

                        future_chart = future.set_index(
                            "timestamp"
                        )["close"]

                        st.line_chart(
                            future_chart
                        )

                        st.dataframe(
                            future[
                                [
                                    "timestamp",
                                    "open",
                                    "high",
                                    "low",
                                    "close"
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                    else:
                        st.info(
                            "No subsequent candles are available "
                            "after this historical match."
                        )


# ==================================================
# 4. Backtesting
# ==================================================

st.divider()
st.header("3. Backtesting")

st.info(
    "A proper walk-forward backtest with performance statistics "
    "has not yet been implemented. Historical similarity scores "
    "alone do not establish whether this method is profitable."
)
