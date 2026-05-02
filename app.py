with tab3:
    st.markdown("## 📄 Smart Report")

    if st.session_state.history:

        last_disease, last_conf = st.session_state.history[-1]

        # ===============================
        # 🧠 SUMMARY
        # ===============================
        st.markdown("### 🧠 Latest Diagnosis")

        col1, col2 = st.columns(2)
        col1.metric("🌱 Disease", last_disease)
        col2.metric("📊 Confidence", f"{last_conf*100:.2f}%")

        # ===============================
        # 🌾 RECOMMENDATION (NEW 🔥)
        # ===============================
        key = last_disease.lower().replace(" ", "_")

        if key in disease_info:
            info = disease_info[key]

            st.markdown("### 🌿 Recommendation")

            st.success(f"Detected: {last_disease}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"📌 **Description:** {info['desc']}")
                st.write(f"⚠ **Cause:** {info['cause']}")

            with col2:
                st.write(f"💊 **Treatment:** {info['treatment']}")
                st.write(f"🌾 **Fertilizer:** {info['fertilizer']}")

        # ===============================
        # 📊 HISTORY CHART (NEW 🔥)
        # ===============================
        st.markdown("### 📊 Prediction Trend")

        diseases = [h[0] for h in st.session_state.history]
        confidences = [h[1]*100 for h in st.session_state.history]

        fig = px.line(
            x=list(range(len(diseases))),
            y=confidences,
            markers=True,
            title="Confidence Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # 📋 CLEAN HISTORY (FIXED)
        # ===============================
        st.markdown("### 📋 Prediction History")

        for i, (d, c) in enumerate(st.session_state.history):
            st.write(f"🔹 {i+1}. **{d}** — {c*100:.2f}%")

    else:
        st.info("No predictions yet. Go to Analysis tab.")
