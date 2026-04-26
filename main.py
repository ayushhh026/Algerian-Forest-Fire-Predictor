from fastapi import FastAPI, Request , Form 
from fastapi.templating import Jinja2Templates
from typing import Annotated
import pickle 
from sklearn.preprocessing import StandardScaler


app = FastAPI()

templates = Jinja2Templates(directory="templates")

## import ridge and scaler 
model=pickle.load(open("models/ridge.pkl",'rb'))
standard_scaler=pickle.load(open("models/scaler.pkl",'rb'))

@app.get("/")
def index(request: Request):## special object of FastApi
    return templates.TemplateResponse("index.html", {"request": request}) ## or return {"message": "Hello"}

@app.get("/predict")
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predict")
def predict(
    request: Request ,
    Temperature: Annotated[float, Form(...)],
    RH: Annotated[float, Form(...)],
    Ws: Annotated[float, Form(...)],
    Rain: Annotated[float, Form(...)],
    FFMC: Annotated[float, Form(...)],
    DMC: Annotated[float, Form(...)],
    ISI: Annotated[float, Form(...)],
    Classes: Annotated[float, Form(...)],
    Region: Annotated[float, Form(...)]                
            ):
    input_data=[[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]

    new_data_scaled=standard_scaler.transform(input_data)

    result=model.predict(new_data_scaled)[0]

    return templates.TemplateResponse("home.html",{"request": request, "result": result})