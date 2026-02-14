import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ==========================================
# 1. GRADE POINT MAPPING (10-Point Scale)
# ==========================================
GRADE_POINTS = {
    'A+': 10,
    'A': 9,
    'B+': 8.25,
    'B': 7.5,
    'C+': 6.75,
    'C': 6,
    'D': 5,
    'F': 0,
    'I': 0,  # Incomplete / Detained
    'Z': 0   # Absent
}

# ==========================================
# 2. CORE GRADING LOGIC (Per Subject)
# ==========================================
class StrictUniversityGrading:
    def __init__(self, total_max_marks, ese_max_marks, course_type, protocol):
        self.M = total_max_marks       
        self.ESE_M = ese_max_marks     
        self.type = course_type
        self.protocol = protocol  
        
        if self.type == 'Practical':
            self.P = 0.50 * self.M 
        else:
            self.P = 0.40 * self.M 

    def process_results(self, df):
        results = df.copy()
        debug_logs = []

        # 1. Attendance Verification
        results['Final_Grade'] = np.where(results['attendance'] < 75, 'I', None)

        # 2. Check for ABSENT (AB) in ESE
        mask_absent = results['ese_marks'].astype(str).str.upper() == 'AB'
        if mask_absent.any():
            results.loc[mask_absent & (results['Final_Grade'].isnull()), 'Final_Grade'] = 'Z'

        # Convert ESE to numeric for checks 
        results['ese_marks_numeric'] = pd.to_numeric(results['ese_marks'], errors='coerce').fillna(0)

        # 3. ESE Minimum Marks Check
        min_ese_threshold = 0.20 * self.ESE_M
        mask_ese_fail = (results['Final_Grade'].isnull()) & (results['ese_marks_numeric'] < min_ese_threshold)
        if mask_ese_fail.any():
            results.loc[mask_ese_fail, 'Final_Grade'] = 'F'

        # 4. Filter Students for Statistics based on PROTOCOL
        if self.protocol == 'Protocol A (Strict)':
            stats_mask = (results['attendance'] >= 75) & (results['ese_marks_numeric'] >= min_ese_threshold)
        else:
            stats_mask = (results['attendance'] >= 75)

        regular_students = results.loc[stats_mask, 'marks'].values
        count = len(regular_students)
        
        # 5. Formula Type Selection
        if count >= 30:
            method = "Relative"
            boundaries = self._calculate_relative_boundaries(regular_students)
        else:
            method = "Absolute"
            boundaries = self._get_absolute_boundaries()

        # 6. Grade Assignment
        mask = results['Final_Grade'].isnull()
        results.loc[mask, 'Final_Grade'] = results.loc[mask, 'marks'].apply(
            lambda x: self._assign_grade(x, boundaries)
        )

        # 7. Assign Grade Points
        results['Grade_Point'] = results['Final_Grade'].map(GRADE_POINTS)
        
        results = results.drop(columns=['ese_marks_numeric'])
        return results, method

    def _calculate_relative_boundaries(self, marks):
        X = np.mean(marks) 
        sigma = np.std(marks)
        raw_D_limit = X - 1.5 * sigma
        
        bounds = {
            'A+': X + 1.5 * sigma,
            'A':  X + 1.0 * sigma,
            'B+': X + 0.5 * sigma,
            'B':  X,
            'C+': X - 0.5 * sigma,
            'C':  X - 1.0 * sigma,
            'D':  raw_D_limit
        }

        if raw_D_limit > self.P:
            bounds['C+'] = X - (1 * (X - self.P) / 3)
            bounds['C']  = X - (2 * (X - self.P) / 3)
            bounds['D']  = float(self.P) 
        elif raw_D_limit < (0.30 * self.M):
            delta = (0.30 * self.M) - raw_D_limit
            for g in bounds:
                bounds[g] += delta
            bounds['D'] = 0.30 * self.M

        if bounds['A+'] > self.M:
             bounds['A+'] = self.M
             
        return bounds

    def _get_absolute_boundaries(self):
        if self.type == 'Theory':
            return {'A+': 90, 'A': 80, 'B+': 72, 'B': 64, 'C+': 56, 'C': 48, 'D': 40}
        else: 
            return {'A+': 90, 'A': 80, 'B+': 70, 'B': 62, 'C+': 58, 'C': 54, 'D': 50}

    def _assign_grade(self, marks, bounds):
        if marks >= bounds['A+']: return 'A+'
        if marks >= bounds['A']:  return 'A'
        if marks >= bounds['B+']: return 'B+'
        if marks >= bounds['B']:  return 'B'
        if marks >= bounds['C+']: return 'C+'
        if marks >= bounds['C']:  return 'C'
        if marks >= bounds['D']:  return 'D'
        return 'F'

# ==========================================
# 3. STREAMLIT WEB INTERFACE
# ==========================================
def generate_sample_csv():
    """Generates a sample CSV with 5 students taking 11 subjects."""
    students = ['CE25001', 'CE25002', 'CE25003', 'CE25004', 'CE25005']
    subjects = [
        ('CE101', 'Maths-III', 3, 'Theory', 100, 60),
        ('CE102', 'Fluid Mechanics', 4, 'Theory', 100, 60),
        ('CE103', 'Solid Mechanics', 4, 'Theory', 100, 60),
        ('CE104', 'Surveying', 3, 'Theory', 100, 60),
        ('CE105', 'Building Materials', 3, 'Theory', 100, 60),
        ('CE106', 'Environmental Engg', 3, 'Theory', 100, 60),
        ('CE107', 'Fluid Lab', 1, 'Practical', 50, 30),
        ('CE108', 'Solid Lab', 1, 'Practical', 50, 30),
        ('CE109', 'Surveying Lab', 1, 'Practical', 50, 30),
        ('CE110', 'AutoCAD Lab', 2, 'Practical', 50, 30),
        ('CE111', 'Seminar', 1, 'Practical', 50, 30)
    ]
    
    data = []
    for sid in students:
        for sub in subjects:
            # Simulate some realistic random data
            att = np.random.randint(70, 100)
            marks = np.random.randint(20, sub[4])
            ese = int(marks * 0.6) if marks > 30 else (np.random.choice(['AB', 10, 5]) if np.random.rand()>0.8 else marks)
            
            data.append([sid, sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], marks, ese, att])
            
    df = pd.DataFrame(data, columns=['student_id', 'subject_code', 'subject_name', 'credits', 'course_type', 'total_max', 'ese_max', 'marks', 'ese_marks', 'attendance'])
    return df.to_csv(index=False).encode('utf-8')

def calculate_sgpa(student_df):
    """Calculates SGPA, Earned Credits, and Failed Subjects for a single student."""
    total_credits = student_df['credits'].sum()
    
    # Earned credits are only for passed subjects
    passed_mask = ~student_df['Final_Grade'].isin(['F', 'I', 'Z'])
    earned_credits = student_df.loc[passed_mask, 'credits'].sum()
    
    # SGPA Calculation = Sum(Credits * Grade Point) / Total Credits
    total_points = (student_df['credits'] * student_df['Grade_Point']).sum()
    sgpa = total_points / total_credits if total_credits > 0 else 0
    
    # Failed Subjects
    failed_subs = student_df.loc[~passed_mask, 'subject_code'].tolist()
    failed_str = ", ".join(failed_subs) if failed_subs else "None"
    
    return pd.Series({
        'Total_Registered_Credits': total_credits,
        'Earned_Credits': earned_credits,
        'SGPA': round(sgpa, 2),
        'Failed_Subjects': failed_str
    })

def main():
    st.set_page_config(page_title="Semester Grading Engine", layout="wide")
    
    st.title("🎓 Multi-Subject Semester Grading Engine")
    st.markdown("**Compliance:** *Tabulation Manual 2026 | Supports SGPA & Multiple Subjects*")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📊 Statistical Protocol")
        protocol_choice = st.radio(
            "Select Grading Logic:",
            ["Protocol A (Strict)", "Protocol B (Inclusive)"],
            help="Protocol A: Excludes ESE failures. Protocol B: Includes failures."
        )
        
        st.markdown("---")
        st.markdown("### 📥 1. Get Template")
        st.download_button(
            label="Download 11-Subject Template CSV", 
            data=generate_sample_csv(), 
            file_name="semester_11_subjects_template.csv", 
            mime="text/csv",
            help="Downloads a pre-filled template with 11 subjects (Theory + Practical)."
        )

        st.markdown("### 📤 2. Upload Data")
        uploaded_file = st.file_uploader("Upload Master CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = {'student_id', 'subject_code', 'subject_name', 'credits', 'course_type', 'total_max', 'ese_max', 'marks', 'ese_marks', 'attendance'}
            
            if not required_cols.issubset(set(df.columns)):
                st.error(f"Error: CSV is missing columns. Required columns: {list(required_cols)}")
            else:
                st.info("🔄 Processing Multiple Subjects...")
                
                # 1. Process Grading Subject by Subject
                all_graded_results = []
                subject_methods = {}
                
                for subject_code, sub_df in df.groupby('subject_code'):
                    # Extract subject rules from the first row of this group
                    total_max = sub_df['total_max'].iloc[0]
                    ese_max = sub_df['ese_max'].iloc[0]
                    course_type = sub_df['course_type'].iloc[0]
                    
                    engine = StrictUniversityGrading(
                        total_max_marks=total_max, 
                        ese_max_marks=ese_max, 
                        course_type=course_type,
                        protocol=protocol_choice
                    )
                    graded_df, method = engine.process_results(sub_df)
                    all_graded_results.append(graded_df)
                    subject_methods[subject_code] = method
                
                # Combine all graded subjects into one master DataFrame
                final_results_df = pd.concat(all_graded_results, ignore_index=True)
                
                # 2. Calculate SGPA Per Student
                sgpa_df = final_results_df.groupby('student_id').apply(calculate_sgpa).reset_index()
                
                # 3. Create Master Result Sheet (Wide Format for Committee)
                # Pivot to get Subject1_Grade, Subject2_Grade... columns
                pivot_grades = final_results_df.pivot(index='student_id', columns='subject_code', values='Final_Grade').reset_index()
                
                # Merge SGPA data with the Pivot table
                master_sheet = pd.merge(pivot_grades, sgpa_df, on='student_id')
                
                # --- DISPLAY RESULTS ---
                st.success(f"✅ Successfully processed {len(df['subject_code'].unique())} subjects for {len(sgpa_df)} students.")
                
                tab1, tab2, tab3 = st.tabs(["📑 Master Result Sheet (SGPA)", "🔍 Detailed Subject View", "📈 Class Statistics"])
                
                with tab1:
                    st.subheader("Semester Grade Report & SGPA")
                    
                    # Highlight Failed rows
                    def highlight_sgpa(row):
                        if "None" not in row['Failed_Subjects']:
                            return ['background-color: #ffcccc'] * len(row)
                        return [''] * len(row)
                        
                    st.dataframe(master_sheet.style.apply(highlight_sgpa, axis=1), use_container_width=True)
                    
                    csv_master = master_sheet.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Master SGPA Sheet", csv_master, 'master_sgpa_results.csv', 'text/csv')

                with tab2:
                    st.subheader("Raw Data with Grade Points")
                    st.dataframe(final_results_df, use_container_width=True)
                    
                    csv_detailed = final_results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Detailed Records", csv_detailed, 'detailed_subject_results.csv', 'text/csv')
                
                with tab3:
                    st.subheader("Subject-wise Processing Log")
                    log_data = [{"Subject": k, "Algorithm Used": v} for k, v in subject_methods.items()]
                    st.table(pd.DataFrame(log_data))
                    
                    st.subheader("SGPA Distribution")
                    chart = alt.Chart(sgpa_df).mark_bar(color='teal', opacity=0.7).encode(
                        x=alt.X('SGPA', bin=alt.Bin(maxbins=10), title="SGPA Range"),
                        y=alt.Y('count()', title="Number of Students"),
                        tooltip=['count()']
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("👋 Download the Template on the left, add your data, and upload it to begin.")

if __name__ == "__main__":
    main()
      
