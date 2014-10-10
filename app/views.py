from app import app

@app.route('/')
    return "Hello World"


@app.route('render')
def render():
    return "Hello World Render"
