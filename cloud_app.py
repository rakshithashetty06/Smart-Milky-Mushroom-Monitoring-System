import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Login Page
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if (
            username == "admin"
            and password == "admin123"
        ):
            st.session_state.logged_in = True
            st.rerun()

        else:
            st.error(
                "Invalid Username or Password"
            )

    st.stop()

# Load Model
model = YOLO("best.pt")

# Title
st.title("🍄 Smart Milky Mushroom Monitoring System")
st.caption("AI + Smart Farming Monitoring System")

st.info(
    "This system uses YOLOv11 and environmental parameters to detect "
    "Healthy, Contaminated and PoorGrowth mushrooms and provide recommendations."
)

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["🔍 Detection", "📊 Analytics", "📜 History"]
)

# ==========================
# TAB 1 : DETECTION
# ==========================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload Mushroom Image",
        type=["jpg", "jpeg", "png"]
    )

    temperature = st.slider(
        "Temperature (°C)",
        10,
        50,
        30
    )

    humidity = st.slider(
        "Humidity (%)",
        0,
        100,
        80
    )

    light_intensity = st.slider(
        "Light Intensity (%)",
        0,
        100,
        50
    )

    ventilation = st.selectbox(
        "Ventilation Status",
        ["Good", "Moderate", "Poor"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Analyze"):

            results = model.predict(
                image,
                conf=0.7,
                verbose=False
            )

            if len(results[0].boxes) == 0:

                st.error(
                    "❌ No mushroom detected."
                )

            else:

                best_idx = results[0].boxes.conf.argmax()

                conf = float(
                    results[0].boxes.conf[best_idx]
                )

                label = int(
                    results[0].boxes.cls[best_idx]
                )

                predicted_class = model.names[label]

                st.success(
                    f"Prediction: {predicted_class}"
                )

                st.write(
                    f"Confidence Score: {conf:.2f}"
                )
                if predicted_class == "Contaminated":
                    st.error("🚨 ALERT: Contamination Detected!")
                elif predicted_class == "PoorGrowth":
                    st.warning("⚠ ALERT: Poor Growth Detected!")
                else:
                    st.success("✅ Healthy Mushroom Detected!")


                # Recommendations

                if predicted_class == "Contaminated":

                    suggestion = f"""
❌ Contamination Detected

Possible Causes:
• Green Mold (Trichoderma)
• Black Mold
• Bacterial Contamination
• Poor Hygiene
• Improper Sterilization
• Contaminated Spawn
• Excess Moisture
• Poor Ventilation

Current Conditions:
• Temperature: {temperature}°C
• Humidity: {humidity}%
• Ventilation: {ventilation}

Possible Effects:
• Contamination Spread
• Reduced Yield
• Slow Mycelium Growth
• Crop Loss

Recommended Action:
• Remove infected bag immediately
• Isolate contaminated bags
• Improve hygiene practices
• Disinfect grow room and tools
• Check substrate sterilization
• Improve ventilation
• Monitor humidity levels
• Use quality spawn
"""

                elif predicted_class == "PoorGrowth":

                    suggestion = f"""
Poor Growth Detected

Possible Causes:
• Low Humidity (<80%)
• Poor Ventilation
• Improper Watering
• Poor Quality Paddy Straw
• Temperature Stress
• Low Light Exposure
• Delayed Spawn Run
• Poor Spawn Quality

Current Conditions:
• Temperature: {temperature}°C
• Humidity: {humidity}%
• Ventilation: {ventilation}

Possible Effects:
• Slow Mycelium Growth
• Delayed Fruiting
• Reduced Yield
• Weak Mushroom Development

Recommended Action:
• Maintain humidity between 80–90%
• Maintain temperature between 25–35°C
• Improve ventilation
• Check substrate quality
• Ensure proper watering
• Use quality spawn
• Monitor daily for contamination
"""

                else:

                    if (
                        25 <= temperature <= 35
                        and 80 <= humidity <= 90
                        and ventilation == "Good"
                    ):

                        suggestion = """
✅ Healthy Mushroom Detected

Status:
• Normal Growth
• No Visible Contamination
• Good Mycelium Development
• Environmental Conditions are Optimal

Recommendation:
• Continue regular monitoring
• Maintain hygiene practices
"""

                    else:

                        suggestion = f"""
⚠ Healthy Mushroom Detected

Environmental Conditions Need Improvement

Current Conditions:
• Temperature: {temperature}°C
• Humidity: {humidity}%
• Ventilation: {ventilation}

Possible Risks:
• Slow Growth
• Delayed Fruiting
• Future Contamination Risk

Recommended Action:
• Maintain temperature between 25–35°C
• Maintain humidity between 80–90%
• Improve ventilation if needed
• Continue monitoring daily
"""

                st.info(suggestion)

                annotated_img = results[0].plot()

                st.image(
                    annotated_img,
                    caption="Detection Result",
                    use_container_width=True
                )

                # Save Data

                data = pd.DataFrame({

                    "Date": [
                        datetime.now()
                    ],

                    "Temperature": [
                        temperature
                    ],

                    "Humidity": [
                        humidity
                    ],

                    "Light": [
                        light_intensity
                    ],

                    "Ventilation": [
                        ventilation
                    ],

                    "Prediction": [
                        predicted_class
                    ],

                    "Confidence": [
                        round(conf, 2)
                    ]
                })

                if os.path.exists(
                    "mushroom_data.csv"
                ):

                    data.to_csv(
                        "mushroom_data.csv",
                        mode="a",
                        header=False,
                        index=False
                    )

                else:

                    data.to_csv(
                        "mushroom_data.csv",
                        index=False
                    )

# ==========================
# TAB 2 : ANALYTICS
# ==========================

with tab2:

    st.header(
        "Analytics Dashboard"
    )

    if os.path.exists(
        "mushroom_data.csv"
    ):

        df = pd.read_csv(
            "mushroom_data.csv"
        )

        counts = (
            df["Prediction"]
            .value_counts()
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Healthy",
                counts.get("Healthy", 0)
            )

        with col2:
            st.metric(
                "Contaminated",
                counts.get("Contaminated", 0)
            )

        with col3:
            st.metric(
                "PoorGrowth",
                counts.get("PoorGrowth", 0)
            )

        st.subheader(
            "Prediction Counts"
        )

        st.bar_chart(counts)

        fig, ax = plt.subplots()

        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

# ==========================
# TAB 3 : HISTORY
# ==========================

with tab3:

    st.header(
        "Prediction History"
    )

    if os.path.exists(
        "mushroom_data.csv"
    ):

        df = pd.read_csv(
            "mushroom_data.csv"
        )

        st.dataframe(df)

        st.download_button(
            label="📥 Download CSV Report",
            data=df.to_csv(index=False),
            file_name="mushroom_report.csv",
            mime="text/csv"
        )
