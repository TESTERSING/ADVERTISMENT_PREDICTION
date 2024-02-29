import streamlit as st
import pandas as pd
import xgboost as xgb


def main():
    html_temp = """
    <div style="background-color: Lightblue; padding: 16px;">
        <h2 style="color: black; text-align: center;">Advertisment Prediction Using ML</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Load the XGBoost model
    model = xgb.XGBRegressor()
    model.load_model('xgb_model.json')  # Replace 'xgb_model.json' with your model file path

    p1 = st.number_input("Enter the channel code", 1, step=1)
    p2 = st.number_input("Enter the programme code", 1, step=1)
    p3 = st.number_input("Enter the programme day ", 1, step=1)
    p4 = st.number_input("Enter the programme month ", 1, step=1)
    p5 = st.number_input("Enter the programme year ", 1, step=1)
    p6 = st.number_input("Enter the programme start hours", 0, step=1)
    p7 = st.number_input("Enter the programme start minutes", 0, step=1)
    p8 = st.number_input("Enter the programme end hours", 0, step=1)
    p9 = st.number_input("Enter the programme end minutes", 0, step=1)

    data_new = pd.DataFrame({
        'channel_code': p1,
        'programe_code': p2,
        'programe_day': p3,
        'programe_month': p4,
        'programe_year': p5,
        'start_hour': p6,
        'start_minute': p7,
        'end_hour': p8,
        'end_minute': p9
    }, index=[0])

    if st.button("predict"):
        pred = model.predict(data_new)
        st.balloons()
        st.success("Predicted advertisement are {}".format(pred))


if __name__ == "__main__":
    main()
