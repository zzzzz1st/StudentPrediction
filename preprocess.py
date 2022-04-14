import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Read CSV FILES
assessments = pd.read_csv("assessments.csv", usecols=["id_assessment", "assessment_type", "weight"])
st_assessments = pd.read_csv("studentAssessment.csv", usecols=["id_assessment", "id_student", "score"])
st_info = pd.read_csv("studentInfo.csv",
                      usecols=["id_student", "highest_education", "num_of_prev_attempts", "studied_credits",
                               "disability", "final_result"])
st_vle = pd.read_csv("studentVle.csv", usecols=["id_student", "sum_click"])

# Change DTYPES IN ASSESSMENTS
assessments.id_assessment = assessments.id_assessment.astype("int32")
assessments.weight = assessments.weight.astype("int32")
# Change DTYPES IN ST_ASSESSMENTS
st_assessments.id_assessment = st_assessments.id_assessment.astype("int32")
st_assessments.id_student = st_assessments.id_student.astype("int32")
# Change DTYPES IN ST_INFO
st_info.id_student = st_info.id_student.astype("int32")
st_info.num_of_prev_attempts = st_info.num_of_prev_attempts.astype("int32")
st_info.studied_credits = st_info.studied_credits.astype("int32")
# Change DTYPES IN ST_VLE
st_vle.id_student = st_vle.id_student.astype("int32")
st_vle.sum_click = st_vle.sum_click.astype("int32")
# TOTAL NUMBER OF STUDENTS
st_info = st_info.sample(1000)
# MERGE THE DATA INTO A UNIQUE DATAFRAME
assJoined = pd.merge(assessments, st_assessments, on="id_assessment")
student = pd.merge(assJoined, st_vle, on="id_student")
student = pd.merge(student, st_info, on="id_student")
# DROP ID WHICH IS NOT NECESSARY
student = student.drop(columns=["id_assessment", "id_student"])
# ENCODERS FOR CATEGORICAL VALUES
assessments_encoder = OrdinalEncoder(categories=[["CMA", "Exam", "TMA"]])

edu_encoder = OrdinalEncoder(categories=[
    ["A Level or Equivalent", "HE Qualification", "Lower Than A Level", "No Formal quals",
     "Post Graduate Qualification"]])
disability_encoder = OrdinalEncoder(categories=[["N", "Y"]])
final_encoder = LabelEncoder()
# ORDINAL ENCODING FOR FEATURE VARIABLES
student["assessment_type"] = assessments_encoder.fit_transform(student[["assessment_type"]])
student["highest_education"] = edu_encoder.fit_transform(student[["highest_education"]])
student["disability"] = disability_encoder.fit_transform(student[["disability"]])
# DIVIDE THE MAIN STUDENT INTO FOUR CASES
studentpf = student[(student.final_result != "Withdrawn") & (student.final_result != "Distinction")]
studentdf = student[(student.final_result != "Withdrawn") & (student.final_result != "Pass")]
studentdp = student[(student.final_result != "Withdrawn") & (student.final_result != "Fail")]
studentwp = student[(student.final_result != "Distinction") & (student.final_result != "Fail")]
# DROP ROWS CONTAINING NaN
studentpf = studentpf.dropna()
studentdf = studentdf.dropna()
studentdp = studentdp.dropna()
studentwp = studentwp.dropna()
# ENCODE LABEL IN ORDER TO HAVE BINARY LABELS
studentpf["final_result"] = final_encoder.fit_transform(studentpf["final_result"])
studentdf["final_result"] = final_encoder.fit_transform(studentdf["final_result"])
studentdp["final_result"] = final_encoder.fit_transform(studentdp["final_result"])
studentwp["final_result"] = final_encoder.fit_transform(studentwp["final_result"])
# DATA SUMMARY
print("SAMPLE DATA FINAL RESULT SUMMARY : ")
print(studentpf["final_result"].value_counts())
print(studentdf["final_result"].value_counts())
print(studentdp["final_result"].value_counts())
print(studentwp["final_result"].value_counts())
# SAVE DATA FRAMES
studentpf.to_pickle("studentpf.pkl")
studentdf.to_pickle("studentdf.pkl")
studentdp.to_pickle("studentdp.pkl")
studentwp.to_pickle("studentwp.pkl")
