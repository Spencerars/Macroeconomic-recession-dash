**Macroeconomic Recession Probability Dashboard (Python / ML)**



An interactive macroeconomic scenario simulator that estimates U.S. recession probability using machine learning and live economic data. The application pulls real-time indicators from FRED and Yahoo Finance, trains a neural network model, and allows users to stress-test macro conditions via a desktop GUI.



Development was accelerated using AI-assisted coding tools, with all system design, modeling decisions, data validation, and integration handled by me. AI was used as a productivity aid for drafting components and iterating quickly, while I reviewed, modified, and finalized all logic to ensure correctness, performance, and interpretability.



**Core Features**



Live economic data ingestion

Automatically fetches and updates macro indicators from the FRED API and S\&P 500 data via Yahoo Finance, with local caching to reduce API calls and improve startup performance.



Machine learning recession classifier

Neural network (MLPClassifier) trained on historical macroeconomic indicators with feature standardization, reproducible configuration, and probability-based output.



Scenario analysis \& stress testing

Users can manually adjust macro indicators to simulate hypothetical economic conditions and observe resulting changes in recession probability.



Statistical context \& interpretability

Displays historical mean, 10th percentile, and 90th percentile values for each feature, providing users with context for how extreme or typical each input is.



Indicator relevance analysis

Computes R² values between individual indicators and recession outcomes to highlight which features historically explain the most variance.



Responsive desktop GUI

Built with Tkinter and Matplotlib, featuring multithreaded data loading, embedded visualizations, and real-time probability updates.



**Technical Highlights**



Language: Python



Libraries:

pandas, numpy

scikit-learn

yfinance, fredapi

tkinter, matplotlib



Engineering concepts:

End-to-end ML pipelines (data ingestion → feature engineering → modeling → inference)

Model validation and probability calibration

Local disk caching and API rate-limit mitigation

Multithreaded UI design

Clean separation between backend model logic and frontend presentation



Tooling:

AI-assisted development for rapid prototyping and iteration

Manual code review, refactoring, and testing for production-quality behavior

My Role \& Contributions

Architected the overall system design and data pipeline

Selected and configured model architecture and features

Integrated live APIs, caching, and persistence

Built the desktop GUI and visualization layer

Used AI tools to accelerate development, then validated, refactored, and finalized all logic



**How to Run**

Run the provided executable or

Clone the repository and run python app.py after installing dependencies

