{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9c0ef03-97cf-41e0-b61e-9fa58706839d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# app.py\n",
    "\n",
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "# === Load the trained Random Forest model pipeline ===\n",
    "model = joblib.load(\"random_forest_pipeline.pkl\")\n",
    "\n",
    "# === App Title ===\n",
    "st.title(\"🍷 Wine Quality Predictor\")\n",
    "st.subheader(\"Powered by Random Forest Classifier\")\n",
    "\n",
    "st.markdown(\"\"\"\n",
    "Use the sliders below to simulate wine properties and predict whether it will be **high-quality** or **low-quality**.\n",
    "\"\"\")\n",
    "\n",
    "# === User Inputs ===\n",
    "alcohol = st.slider(\"Alcohol (%)\", 8.0, 15.0, 11.0, step=0.1)\n",
    "density = st.slider(\"Density (g/cm³)\", 0.9900, 1.0050, 0.9950, step=0.0001)\n",
    "volatile_acidity = st.slider(\"Volatile Acidity (g/dm³)\", 0.10, 1.50, 0.40, step=0.01)\n",
    "\n",
    "input_data = pd.DataFrame([[alcohol, density, volatile_acidity]],\n",
    "                          columns=[\"alcohol\", \"density\", \"volatile acidity\"])\n",
    "\n",
    "# === Predict Button ===\n",
    "if st.button(\"Predict Wine Quality\"):\n",
    "    prediction = model.predict(input_data)[0]\n",
    "    probability = model.predict_proba(input_data)[0]\n",
    "\n",
    "    quality_label = \"🍇 High Quality\" if prediction == 1 else \"🍷 Low Quality\"\n",
    "    confidence = round(np.max(probability) * 100, 2)\n",
    "\n",
    "    st.markdown(f\"### Prediction: **{quality_label}**\")\n",
    "    st.markdown(f\"Confidence: **{confidence}%**\")\n",
    "\n",
    "    st.markdown(\"### Input Summary\")\n",
    "    st.dataframe(input_data.style.format(\"{:.4f}\"))\n",
    "\n",
    "st.markdown(\"---\")\n",
    "st.caption(\"Model trained on physicochemical attributes of red and white wines.\\nPrediction uses alcohol, density, and volatile acidity.\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
