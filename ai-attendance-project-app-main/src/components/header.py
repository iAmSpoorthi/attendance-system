import streamlit as st
import os
import base64


def header_home():
    logo_path = os.path.join("src", "assets", "attendiq_logo.png")

    # Convert logo to base64 so it can be displayed cleanly in HTML
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            margin-bottom: 35px;
        ">
            <img
                src="data:image/png;base64,{logo_base64}"
                style="
                    width: 420px;
                    max-width: 80%;
                    height: auto;
                    display: block;
                "
            >
        </div>
        """,
        unsafe_allow_html=True
    )

def header_dashboard():
    logo_path = os.path.join("src", "assets", "attendiq_logo.png")

    import base64

    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
            margin-bottom: 20px;
        ">
            <img
                src="data:image/png;base64,{logo_base64}"
                style="
                    width: 300px;
                    height: auto;
                    display: block;
                "
            >
        </div>
        """,
        unsafe_allow_html=True
    )