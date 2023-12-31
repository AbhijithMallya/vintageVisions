from flask import Flask , render_template , request
import pickle
import numpy as np
app = Flask(__name__ , static_url_path='/static')



@app.route("/" , methods=['GET','POST'])
def home():
    if request.method == 'POST':
        # Process the form data when the form is submitted
        fixedAcidity = request.form.get('fixedAcidity')
        volatileAcidity = request.form.get('volatileAcidity')
        citricAcid = request.form.get('citricAcid')
        residualSugar = request.form.get('residualSugar')
        chlorides = request.form.get('chlorides')
        freeSulphur = request.form.get('freeSulphur')
        totalSulphur = request.form.get('totalSulphur')
        density = request.form.get('density')
        pH = request.form.get('pH')
        sulphates = request.form.get('sulphates')
        alcohol = request.form.get('alcohol')

        # Here, you can perform any logic to handle the form data
        # For now, we print the values to the console
        print(f'Fixed Acidity: {fixedAcidity}')
        print(f'Volatile Acidity: {volatileAcidity}')
        print(f'Citric Acid: {citricAcid}')
        print(f'Residual Sugar: {residualSugar}')
        print(f'Chlorides: {chlorides}')
        print(f'Free Sulphur Dioxide: {freeSulphur}')
        print(f'Total Sulphur Dioxide: {totalSulphur}')
        print(f'Density: {density}')
        print(f'pH: {pH}')
        print(f'Sulphates: {sulphates}')
        print(f'Alcohol: {alcohol}')


        #Load the model
        model = pickle.load(open('WineRegressorModel.pkl', 'rb'))
        print("Model loaded successfully")

        input = [fixedAcidity,volatileAcidity,citricAcid,residualSugar,chlorides,freeSulphur,totalSulphur,density,pH,sulphates,alcohol]
        print("Input Array : ",input)
        input_features = np.array(input)
        input_features_reshaped = input_features.reshape(1, -1)
        result = model.predict(input_features_reshaped)
        print( "Wine Quality : ",result)
        result = "Wine Quality : "+str(result)
        
        return render_template('index.html' , wineQuality=str(result))


   
    return render_template('index.html')
