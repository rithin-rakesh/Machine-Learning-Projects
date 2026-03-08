import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD MODEL + ENCODERS ----------------
@st.cache_resource
def load_files():
    model = pickle.load(open("XGModel.sav", "rb"))

    ohn = pickle.load(open("Occupation_Onehot.sav", "rb"))
    ohn1 = pickle.load(open("Marital_status_Onehot.sav", "rb"))
    ohn2 = pickle.load(open("Residential_st_Onehot.sav", "rb"))
    ohn3 = pickle.load(open("PurposeofLoan_Onehot.sav", "rb"))
    ohn4 = pickle.load(open("Collateral_Onehot.sav", "rb"))
    ohn5 = pickle.load(open("LocationofApplication_Onehot.sav", "rb"))
    ohn6 = pickle.load(open("DeviceInfo_Onehot.sav", "rb"))
    ohn7 = pickle.load(open("Refferal_Onehot.sav", "rb"))

    return model, ohn, ohn1, ohn2, ohn3, ohn4, ohn5, ohn6, ohn7


model, ohn, ohn1, ohn2, ohn3, ohn4, ohn5, ohn6, ohn7 = load_files()


# ---------------- MAIN APP ----------------
def main():

    st.title("🔍 Loan Fraud Detection System")
    st.markdown("---")

    # ---------------- NUMERICAL INPUTS ----------------
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100)
        AddressDuration = st.number_input("Address Duration")
        Dependents = st.number_input("Dependents")
        IncomeLevel = st.number_input("Income Level")

    with col2:
        ExistingLiabilities = st.number_input("Existing Liabilities")
        InterestRate = st.number_input("Interest Rate")
        LoanTerm = st.number_input("Loan Term")
        SocialMediaFootprint = st.selectbox("Social Media Footprint", [0, 1])

    st.markdown("### Applicant Categorical Details")

    # ---------------- SELECTBOXES ----------------
    Occupation = st.selectbox("Occupation", ohn.categories_[0])
    MaritalStatus = st.selectbox("Marital Status", ohn1.categories_[0])
    ResidentialStatus = st.selectbox("Residential Status", ohn2.categories_[0])
    PurposeoftheLoan = st.selectbox("Purpose of Loan", ohn3.categories_[0])
    Collateral = st.selectbox("Collateral", ohn4.categories_[0])
    LocationofApplication = st.selectbox("Location of Application", ohn5.categories_[0])
    DeviceInformation = st.selectbox("Device Information", ohn6.categories_[0])
    Referral = st.selectbox("Referral Source", ohn7.categories_[0])

    st.markdown("---")

    if st.button("🚀 Predict Fraud"):

        try:
            # ---------------- BASE NUMERIC DF ----------------
            base_df = pd.DataFrame([[
                Age, AddressDuration, Dependents, IncomeLevel,
                ExistingLiabilities, InterestRate,
                LoanTerm, SocialMediaFootprint
            ]], columns=[
                "Age","AddressDuration","Dependents","IncomeLevel",
                "ExistingLiabilities","InterestRate",
                "LoanTerm","SocialMediaFootprint"
            ])

            # ---------------- ENCODING ----------------
            occ = pd.DataFrame(
                ohn.transform([[Occupation]]).toarray(),
                columns=ohn.get_feature_names_out()
            )

            mar = pd.DataFrame(
                ohn1.transform([[MaritalStatus]]).toarray(),
                columns=ohn1.get_feature_names_out()
            )

            res = pd.DataFrame(
                ohn2.transform([[ResidentialStatus]]).toarray(),
                columns=ohn2.get_feature_names_out()
            )

            pur = pd.DataFrame(
                ohn3.transform([[PurposeoftheLoan]]).toarray(),
                columns=ohn3.get_feature_names_out()
            )

            col = pd.DataFrame(
                ohn4.transform([[Collateral]]).toarray(),
                columns=ohn4.get_feature_names_out()
            )

            loc = pd.DataFrame(
                ohn5.transform([[LocationofApplication]]).toarray(),
                columns=ohn5.get_feature_names_out()
            )

            dev = pd.DataFrame(
                ohn6.transform([[DeviceInformation]]).toarray(),
                columns=ohn6.get_feature_names_out()
            )

            ref = pd.DataFrame(
                ohn7.transform([[Referral]]).toarray(),
                columns=ohn7.get_feature_names_out()
            )

            # ---------------- CONCAT ALL ----------------
            final_df = pd.concat(
                [base_df, occ, mar, res, pur, col, loc, dev, ref],
                axis=1
            )

            # 🔥 CRITICAL FIX (COLUMN MATCHING)
            final_df = final_df.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

            # ---------------- PREDICTION ----------------
            prediction = model.predict(final_df)
            probability = model.predict_proba(final_df)

            st.markdown("## 🔎 Result")

            if prediction[0] == 1:
                st.error("🚨 Fraudulent Application Detected")
            else:
                st.success("✅ Genuine Application")

            st.info(f"Fraud Probability: {round(probability[0][1]*100,2)} %")

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()