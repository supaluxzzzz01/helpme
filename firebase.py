import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, request, render_template, redirect, url_for

# Initialize Firebase Admin SDK
cred = credentials.Certificate('/Users/supaluxnavarattanasiri/Desktop/neverfall/key.json')
firebase_admin.initialize_app(cred)

# Get a reference to the Firestore database
db = firestore.client()

#flask
app = Flask(__name__)

# Route to get all documents from a collection
@app.route('/collection_name', methods=['GET', 'POST'])
def get_documents():
    collection_ref = db.collection('collection_name')
    docs = collection_ref.stream()
    data = []
    for doc in docs:
        data.append(doc.to_dict())
    return render_template('collection_name.html', documents=data)

# Route to create a new document in a collection
@app.route('/collection_name', methods=['POST'])
def create_document():
    data = request.json
    collection_ref = db.collection('collection_name')
    collection_ref.add(data)
    return jsonify({'message': 'Document created successfully'})

@app.route('/add_name', methods=['GET'])
def add_name():
    name = request.form.get('name')  # Read the name from the form data
    doc_ref = db.collection('names').document()  # Create a new document in the 'names' collection
    doc_ref.set({'name': name})  # Set the 'name' field to the value of the 'name' variable
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)