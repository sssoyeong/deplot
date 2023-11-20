import plotly.express as px

df = px.data.gapminder().query("continent=='Asia'")
# type(df) = pandas.DataFrame

fig = px.line(df, x="year", y="lifeExp", color='country')
fig.update_xaxes(rangeslider_visible=True)
fig.show()