# 🚗 Car Price Prediction using Linear Regression (Flask + Scikit-Learn)

A **Machine Learning web application** built using **Flask**, **Scikit-Learn**, **Pandas**, and **NumPy** that predicts the price of a car based on various features such as mileage, year, fuel type, and more.  
The model is trained using **Linear Regression** to provide accurate and interpretable predictions.

---

## 🚀 Features

- 📈 Predicts car prices based on user input (e.g., brand, model year, fuel type, etc.)  
- 🧠 Uses **Linear Regression** for simple and effective price prediction  
- 🌐 Built with **Flask** for interactive web-based deployment  
- 🧹 Data preprocessing using **Pandas** and **NumPy**  
- 💾 Model saved and loaded using **Pickle** for real-time inference  

---

## 🧰 Tech Stack

| Component | Technology Used |
|------------|-----------------|
| Framework | Flask |
| Language | Python |
| Libraries | Pandas, NumPy, Scikit-learn, Pickle |
| Model | Linear Regression |
| Frontend | HTML, CSS, Bootstrap |
| Deployment | Localhost / Render / Heroku |

---

## 📂 Project Structure

```
car_price_prediction/
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── car_price_prediction.ipynb   # Model training notebook
├── app.py                        # Flask app entry point
├── model.pkl                     # Trained Linear Regression model
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/car-price-prediction.git
   cd car-price-prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate       # On Windows
   source venv/bin/activate    # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**
   ```bash
   python app.py
   ```

5. **Open the app in your browser**
   👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🧠 Model Overview

The **Linear Regression** model was trained on a dataset containing car attributes such as:  
- Year of manufacture  
- Present price  
- KMs driven  
- Fuel type  
- Seller type  
- Transmission  

After preprocessing and feature encoding, the model learns the relationship between these features and the car’s selling price.

---

## 💡 How It Works

1. User enters car details in the web form.  
2. Flask routes the data to the backend.  
3. The trained **Linear Regression** model (loaded from `model.pkl`) predicts the price.  
4. The predicted price is displayed on the result page.

---

## 🧪 Example Prediction

```
Input:
Car Name: Maruti Swift
Year: 2018
Fuel Type: Petrol
KMs Driven: 25,000

Output:
Predicted Price: ₹4.85 Lakh
```

---

## 🔮 Future Enhancements

- 🚘 Integrate multiple regression models (Ridge, Lasso, Random Forest)  
- 📊 Add data visualization dashboard  
- ☁️ Deploy to Render / AWS / Heroku  
- 📱 Build a responsive UI with better styling  

---

## 👨‍💻 Author

**Gaurav Pandwar**  
📧 [gauravpandwar@gmail.com](mailto:gauravpandwar@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/gp30) | [GitHub](https://github.com/gaurav-pandwar)

---

## 🪪 License

This project is licensed under the **MIT License** — you’re free to use and modify it.
