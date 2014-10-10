from app import app

@app.route('/')
def index():
    return "Hello World"


@app.route('/render')
def render():
    return "Hello World Render"
