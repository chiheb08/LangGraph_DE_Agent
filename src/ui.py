import streamlit as st
from agent import MediumAgent

# Initialize the agent
agent = MediumAgent()

# Set the page configuration
st.set_page_config(
    page_title='DE Agent - Medium Article Analyzer',
    page_icon='🔍',
    layout='centered'
)

# Add a header image from the local 'images' directory
st.image('images/data-engineering.png', use_container_width=True)

st.title('DE Agent - Medium Article Analyzer')

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Analyze Medium Articles on Data Engineering</p>', unsafe_allow_html=True)

# URL input for user to enter the article link
article_url = st.text_input('Enter Medium article URL:')

if st.button('Analyze Article'):
    if article_url:
        # Run the agent on the article URL
        result = agent.run(article_url)
        
        if result:
            # Display the results
            st.subheader('Results')
            st.write('**Classification:**', result["classification"])
            st.write('**Entities:**', ', '.join(result["entities"]))
            st.write('**Summary:**', result["summary"])
            st.write('**References:**', ', '.join(result["references"]))
        else:
            st.error('Failed to fetch or analyze the article.')
    else:
        st.error('Please enter a valid URL.') 