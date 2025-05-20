from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
import base64


def generate_wordcloud_base64(palavras):
    color_map = ListedColormap(['orange', 'green', 'red', 'magenta'])
    cloud = WordCloud(
        background_color='white',
        max_words=100,
        colormap=color_map
    ).generate(' '.join(palavras))

    buf = io.BytesIO()
    cloud.to_image().save(buf, format="PNG")
    return {"wordcloud_image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"}