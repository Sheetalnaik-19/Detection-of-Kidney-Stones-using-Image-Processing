from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from kidney_stone_detection import preprocess_image, detect_edges, find_contours, check_for_kidney_stones, draw_contours

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process the image
        image = cv2.imread(filepath)
        preprocessed = preprocess_image(image)
        edges = detect_edges(preprocessed)
        contours = find_contours(edges)

        if check_for_kidney_stones(contours):
            result_message = "Kidney stone detected"
            result_image = draw_contours(image.copy(), contours)
        else:
            result_message = "No kidney stone detected"
            result_image = image

        # Save the result image
        result_path = os.path.join('static', 'output.jpg')
        cv2.imwrite(result_path, result_image)

        return render_template('index.html', result=result_message)

if __name__ == '__main__':
    app.run(debug=True)
