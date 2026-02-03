from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime, timedelta
import sys
from pathlib import Path

def get_app_data_dir(app_name="Full_Dashboard_1225"):
    if sys.platform.startswith("win"):
        base = Path(os.environ["APPDATA"])
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home()
    path = base / app_name
    path.mkdir(parents=True, exist_ok=True)
    return path

# --- BACKEND LOGIC ---

class RecessionModel:
    def __init__(self):
        self.fred = Fred(api_key="4d4b0522a8bd04e74e977fcb8243ea93") 
        self.model = None
        self.scaler = None
        self.latest_data = None
        self.df_columns = None
        # <--- NEW: Store statistics here
        self.feature_stats = {} 
        self.prob_stats = {}

    def get_fred_series_cached(self, series_id, start="1980-01-01", max_age_days=7):
        cache_dir = get_app_data_dir() / "fred_cache"
        cache_dir.mkdir(exist_ok=True)

        cache_file = cache_dir / f"{series_id}.csv"

        if cache_file.exists():
            modified = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - modified < timedelta(days=max_age_days):
                return pd.read_csv(
                    cache_file,
                    index_col=0,
                    parse_dates=True
                ).iloc[:, 0]

        s = self.fred.get_series(series_id, observation_start=start)
        s.to_csv(cache_file)
        return s
    
    def fetch_and_train(self):
        series = {
            "Initial Claims (Unemployment)": "ICSA",
            "New Orders: Consumer Goods": "ACOGNO",
            "New Orders: Nondef Cap Goods ex-Aircraft": "ANDENO",
            "Building Permits": "PERMIT",
            "Interest Rate Spread (10Y - Fed Funds)": "T10YFF",
            "Consumer Expectations": "UMCSENT",
            "Recession": "USRECD"
        }

        data = {}
        for name, sid in series.items():
            try:
                s = self.get_fred_series_cached(sid)
                s = s.resample("ME").mean()
                data[name] = s
            except Exception as e:
                print(f"Error fetching {name}: {e}")

        df = pd.DataFrame(data)

        sp500 = yf.download('^GSPC', start='1980-01-01', interval='1d', progress=False)
        if not sp500.empty:
            sp500 = sp500['Close'].resample('ME').mean()
            if isinstance(sp500.columns, pd.MultiIndex):
                sp500 = sp500.iloc[:, 0]
            
            df["Stock Prices (S&P 500)"] = sp500
            df["Stock Prices (S&P 500)"] = df["Stock Prices (S&P 500)"].pct_change()
        
        df.dropna(inplace=True)
        
        self.df_columns = [c for c in df.columns if c != "Recession"]
        X = df[self.df_columns]
        y = df['Recession']

        # <--- NEW: Calculate Statistics (Mean, 10%, 90%) ---
        # 10% = Low value, 90% = High value
        stats_df = X.describe(percentiles=[0.1, 0.9]).T
        self.feature_stats = stats_df[['mean', '10%', '90%']].to_dict('index')

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = MLPClassifier(
            hidden_layer_sizes=(6,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_scaled, y)

        # <--- NEW: Calculate Average Historical Recession Probability ---
        all_probs = self.model.predict_proba(X_scaled)[:, 1]
        self.prob_stats = {
            'mean': np.mean(all_probs),
            'high_risk': np.percentile(all_probs, 90) # Top 10% risk threshold
        }

        self.latest_data = X.iloc[-1].to_dict()
        
        baseline_prob = self.predict(self.latest_data)
        self.df = df
        return baseline_prob

    def predict(self, inputs):
        input_df = pd.DataFrame([inputs])
        input_scaled = self.scaler.transform(input_df)
        prob = self.model.predict_proba(input_scaled)[0][1]
        return prob
    
    def get_r2(self, n=2):
        results = []
        y = self.df['Recession']

        for feature in self.df_columns:
            X = self.df[[feature]]
            reg = LinearRegression().fit(X, y)
            r2 = reg.score(X, y)
            results.append((feature, r2))
        
        results.sort(key=lambda x: x[1], reverse=True)

        top_features = []
        for feature, r2 in results[:n]:
            top_features.append(
                (feature, r2, self.df[feature], self.df['Recession'])
            )
        return top_features

# --- FRONTEND GUI ---

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Economic Recession Simulator")
        self.root.geometry("1200x800") # Made wider to fit new columns

        self.logic = RecessionModel()
        self.entries = {}

        header = tk.Frame(root)
        header.pack(fill="x", pady=(10, 5))

        lbl_title = tk.Label(header, text="Macroeconomic Scenario Simulator", font=("Helvetica", 16, "bold"))
        lbl_title.pack()

        self.lbl_status = tk.Label(header, text="Status: Initializing...", fg="blue")
        self.lbl_status.pack()

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Adjusted column weights for wider input area
        self.main_frame.columnconfigure(0, weight=3)  
        self.main_frame.columnconfigure(1, weight=2) 

        left_panel = tk.Frame(self.main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.frame_inputs = tk.Frame(left_panel)
        self.frame_inputs.pack(fill="both", expand=True)

        frame_btns = tk.Frame(left_panel)
        frame_btns.pack(pady=10)

        btn_calc = tk.Button(frame_btns, text="Calculate Odds", command=self.calculate, bg="#4CAF50", fg="white", font=("Arial", 12))
        btn_calc.pack(side="left", padx=5)

        btn_reset = tk.Button(frame_btns, text="Reset to Latest", command=self.reset_values)
        btn_reset.pack(side="left", padx=5)

        self.lbl_result = tk.Label(left_panel, text="Recession Odds: --%", font=("Helvetica", 20, "bold"), fg="#333")
        self.lbl_result.pack(pady=10)
        
        # <--- NEW: Label for Historical Context below the result
        self.lbl_context = tk.Label(left_panel, text="", font=("Arial", 10), fg="gray")
        self.lbl_context.pack(pady=0)

        self.right_panel = tk.Frame(self.main_frame, bg="white", relief="sunken", borderwidth=1)
        self.right_panel.grid(row=0, column=1, sticky="nsew")

        threading.Thread(target=self.load_data, daemon=True).start()

    def load_data(self):
        self.lbl_status.config(text="Fetching live data from FRED & Yahoo...")
        try:
            baseline = self.logic.fetch_and_train()
            self.root.after(0, self.create_input_fields)
            self.root.after(0, self.plot_best_indicator)
            self.root.after(0, lambda: self.lbl_status.config(text="Model Ready. Data loaded.", fg="green"))
            self.root.after(0, lambda: self.update_result(baseline))
        except Exception as e:
            self.root.after(0, lambda: self.lbl_status.config(text=f"Error: {str(e)}", fg="red"))

    def create_input_fields(self):
        for widget in self.frame_inputs.winfo_children():
            widget.destroy()

        # <--- NEW: Add Headers for Stats Columns
        headers = ["Indicator", "Input", "Mean", "10% (Low)", "90% (High)"]
        for col, text in enumerate(headers):
            tk.Label(self.frame_inputs, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=5, pady=5)

        row = 1
        for feature in self.logic.df_columns:
            # 1. Name
            tk.Label(self.frame_inputs, text=feature, anchor="w").grid(row=row, column=0, sticky="w", pady=5)
            
            # 2. Input Entry
            val = self.logic.latest_data[feature]
            is_small = "Stock" in feature or "Spread" in feature
            val_str = f"{val:.6f}" if is_small else f"{val:.2f}"

            entry = tk.Entry(self.frame_inputs, width=12)
            entry.insert(0, val_str)
            entry.grid(row=row, column=1, padx=5, pady=5)
            self.entries[feature] = entry

            # 3. Add Stats Labels (Mean, 10%, 90%)
            stats = self.logic.feature_stats.get(feature, {})
            
            # Helper for formatting
            def fmt(v): return f"{v:.6f}" if is_small else f"{v:.2f}"

            tk.Label(self.frame_inputs, text=fmt(stats.get('mean', 0)), fg="gray").grid(row=row, column=2, padx=5)
            tk.Label(self.frame_inputs, text=fmt(stats.get('10%', 0)), fg="gray").grid(row=row, column=3, padx=5)
            tk.Label(self.frame_inputs, text=fmt(stats.get('90%', 0)), fg="gray").grid(row=row, column=4, padx=5)
            
            row += 1

    def plot_best_indicator(self):
        top_features = self.logic.get_r2(n=3)
        for widget in self.right_panel.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(6, 6), dpi=100)
        axes = fig.subplots(3, 1, sharex=True)

        for ax, (name, r2, feature_data, recession_data) in zip(axes, top_features):
            ax.plot(feature_data.index, feature_data.values, linewidth=1.5, label=name)
            ax.set_ylabel(name, fontsize=8)
            ax.fill_between(recession_data.index, 0, recession_data.values * feature_data.max(), alpha=0.2, color='gray', label="Recession")
            ax.set_title(f"{name} (R² = {r2:.3f})", fontsize=10)
        
        fig.autofmt_xdate()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.right_panel)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def calculate(self):
        try:
            user_inputs = {}
            for feat, entry in self.entries.items():
                user_inputs[feat] = float(entry.get())
            
            prob = self.logic.predict(user_inputs)
            self.update_result(prob)
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers.")

    def update_result(self, prob):
        text = f"{prob:.2%}"
        color = "green"
        if prob > 0.15: color = "orange"
        if prob > 0.25: color = "red"
        
        self.lbl_result.config(text=f"Recession Odds: {text}", fg=color)
        
        # <--- NEW: Update context label
        avg_risk = self.logic.prob_stats.get('mean', 0)
        high_risk = self.logic.prob_stats.get('high_risk', 0)
        self.lbl_context.config(
            text=f"(Historical Avg: {avg_risk:.1%} | High Risk Threshold: >{high_risk:.1%})"
        )

    def reset_values(self):
        self.create_input_fields()
        self.calculate()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()