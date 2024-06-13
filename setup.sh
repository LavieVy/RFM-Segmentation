mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"lethongnhut@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
audio = portaudio\n\
enableCORS=false\n\
enableXsrfProtection = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
