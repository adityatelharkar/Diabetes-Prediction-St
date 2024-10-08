import streamlit as st
import pickle
import pandas as pd


# Load the trained model and scaler
model = pickle.load(open("modelForPrediction.pickle", 'rb'))
scaler = pickle.load(open("standardScalar.pickle", 'rb'))


# Define the main function for the Streamlit app
def main():
    st.title("🩺 Diabetes Prediction Application")

    # Introduction text with improved styling using Streamlit's native markdown
    st.markdown("#### 📊 Check Your Diebetes Report")
    st.image("https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/blogs/22606/images/a04bb60-5ec7-be08-2f28-28f5b485077_DALL_E_2024-05-29_21.17.41_-_Create_an_image_sized_1280x720_pixels_with_no_text_or_words_illustrating_an_interactive_LLM-supported_prediction_session_for_diabetes._The_visual_sh.webp")



    # Input form for user data
    with st.form("diabetes_form"):
        st.subheader("**Please enter your health details to make Prediction**")



        pregnancies = st.number_input("***Number of Pregnancies***", min_value=0, max_value=20, step=1, format="%d")
        glucose = st.number_input("***Glucose Level  (mg/dL)***", min_value=0, max_value=300, step=1, format="%d")
        blood_pressure = st.number_input("***Blood Pressure  (mm Hg)***", min_value=0, max_value=200, step=1, format="%d")
        skin_thickness = st.number_input("***Skin Thickness (mm)***", min_value=0, max_value=100, step=1, format="%d")
        insulin = st.number_input("***Insulin Level  (μU/mL)***", min_value=0, max_value=900, step=1, format="%d")
        bmi = st.number_input("***Body Mass Index  (BMI)***", min_value=0.0, max_value=70.0, step=0.1, format="%.1f")
        dpf = st.number_input("***Diabetes Pedigree Function***", min_value=0.0, max_value=2.5, step=0.01, format="%.2f")
        age = st.number_input("***Age (years)***", min_value=10, max_value=120, step=1, format="%d")

        # Submit button
        submitted = st.form_submit_button("***Click to Predict***")

    # If the user submits the form
    if submitted:
        try:
            # Create DataFrame for the inputs
            input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                      columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                               'BMI', 'DiabetesPedigreeFunction', 'Age'])

            # Standardize the input data
            standardized_data = scaler.transform(input_data)

            # Predict using the loaded model
            prediction = model.predict(standardized_data)[0]

            # Display result with conditional formatting for the output
            if prediction == 1:
                st.error("⚠️ Warning: Your Report for Diabetes is :- ***POSITIVE***")
                st.markdown("##### Based on your positive result, it is recommended that you take the following precautions:")
                st.write("1. Regularly monitor blood glucose as advised by your doctor.")
                st.write("2. Take insulin or medications (like Metformin) as prescribed without skipping doses.")
                st.write("3. Stay hydrated to prevent blood sugar fluctuations.")
                st.write("4. Manage stress with meditation, yoga, or deep breathing.")
                st.write("5. Follow a fiber-rich, low-sugar, low-fat diabetic-friendly diet.")

            else:
                st.success("✅ Great! Your Report for Diabetes is :- ***NEGATIVE***")

            st.balloons()

        except Exception as e:
            st.error(f"An error occurred: {e}")



# Entry point for the app
if __name__ == '__main__':
    main()
