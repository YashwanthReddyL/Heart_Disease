from pydantic import BaseModel
from typing import Literal


class HeartData(BaseModel):
    age: int
    sex: Literal["Male", "Female"]
    dataset: Literal["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    cp: Literal[
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ]
    trestbps: int
    chol: int
    fbs: bool
    restecg: Literal["normal", "lv hypertrophy", "st-t abnormality"]
    thalch: int
    exang: bool
    oldpeak: float
    slope: Literal["upsloping", "flat", "downsloping"]
    ca: int
    thal: Literal["normal", "fixed defect", "reversable defect"]
