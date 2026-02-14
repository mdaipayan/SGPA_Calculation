# 🎓 Multi-Subject Semester Grading & SGPA Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B)
![Compliance](https://img.shields.io/badge/Compliance-UGC%20%7C%20AICTE%20%7C%20NEP%202020-success)

A robust, data-driven web application designed to automate Relative Grading, calculate Semester Grade Point Averages (SGPA), and generate Master Result Sheets for autonomous engineering institutions. 

**Developed by:** Daipayan Mandal (Assistant Professor, Civil Engineering)  
**Compliance:** Tabulation Manual 2026 (KITS Ramtek)

## 🚀 Key Features

* **Full Semester Processing:** Process up to 11+ subjects (Theory & Practical) simultaneously using a Long-Format CSV.
* **Automated SGPA Calculation:** Automatically calculates total registered credits, earned credits, and SGPA on a 10-point scale.
* **Dual Statistical Protocols:**
  * **Protocol A (Strict/OBE Compliant):** Excludes outliers (ESE failures & absentees) from the Mean/SD calculation to prevent grade inflation and maintain high academic standards.
  * **Protocol B (Inclusive):** Includes all students in the statistical baseline (for comparative analysis).
* **Smart Edge-Case Handling:**
  * Detects `AB` (Absent) in End Semester Exams
  * 
