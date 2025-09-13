import streamlit as st
from PIL import Image

def About():
    # Custom Styling
    st.markdown(
        """
        <style>
        body { background-color: #f8f9fa; font-family: Arial, sans-serif; }
        .container { max-width: 800px; margin: auto; padding: 20px; }
        .header { text-align: center; color: #1E3A8A; font-size: 42px; font-weight: bold; }
        .sub-header { font-size: 26px; font-weight: bold; color: #1E40AF; margin-top: 20px; }
        .text { font-size: 18px; color: #374151; line-height: 1.6; }
        .highlight { color: #DC2626; font-weight: bold; }
        .quote { font-size: 20px; font-style: italic; color: #F59E0B; text-align: center; margin-top: 10px; }
        .card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        </style>
        """, unsafe_allow_html=True
    )

    # Profile Header
    st.markdown("""
        <div class='container'>
        <div class='card'>
        <p class='header'>👨‍💻 Prathamesh Jadhav - AI & Data Science Engineer 🚀</p>
        <p class='quote'>‘Innovating with AI, learning every day, and always up for a challenge!’ 🤖💡</p>
        </div>
        """, unsafe_allow_html=True)

    # Summary Section
    st.markdown("""
        <div class='container'>
        <div class='card'>
        <p class='text'>Hello! I'm a passionate <span class='highlight'>B.Tech AI & Data Science Engineer</span> with expertise in Machine Learning, Full-Stack Development, and problem-solving. I love turning ideas into impactful innovations! 🌟</p>
        </div>
        </div>
    """, unsafe_allow_html=True)

    # Education
    st.markdown("""
        <div class='container'>
        <div class='card'>
        <p class='sub-header'>🎓 Education</p>
        <p class='text'>- **B.Tech Artificial Intelligence and Data Science**</p>
        <p class='text'>- *First Rank - B.Tech (AI & DS)*</p>
        <p class='text'>- *Secured 1st rank in academics and 4th prize in state-level Kabaddi tournament in Pune.*</p>
        </div>
        </div>
    """, unsafe_allow_html=True)

    # Skills Section
    st.markdown("""
        <div class='container'>
        <div class='card'>
        <p class='sub-header'>🚀 Skills & Expertise</p>
        <ul class='text'>
        <li><b>Languages:</b> Python, C++, Java, SQL</li>
        <li><b>Frameworks & Libraries:</b> Numpy, Pandas, Scikit-learn, TensorFlow, Keras, Streamlit</li>
        <li><b>Java Full Stack:</b> HTML5, CSS, JavaScript, React.js,Core Java, Spring Boot, Servlet, JSP, JDBC, Git & GitHub, Maven, Microservices, Postman, Docker</li>
        <li><b>Databases & Tools:</b> MySQL, MongoDB, Git, Docker, Flask, Power BI,  IntelliJ IDEA, VS Code</li>
        <li><b>Soft Skills:</b> Problem-solving, Teamwork, Effective Communication</li>
        </ul>
        </div>
        </div>
    """, unsafe_allow_html=True)

    # Projects Section
    st.markdown("""
        <div class='container'>
        <div class='card'>
        <p class='sub-header'>🔬 Projects & Innovations</p>
        <ul class='text'>
        <li><b>⚡ EV Charging Time Prediction & Optimization</b> *(2024 - Present)* - Flask-based ML web app for optimizing EV charging time.</li>
        <li><b>📈 Placement Prediction WebApp</b> *(2024)* - Predicting student placement using ML techniques.</li>
        <li><b>🌱 AgriTech Assist</b> *(2024)* - ML-based crop disease detection and fertilizer recommendations.</li>
        <li><b>📊 Air Quality Index Prediction</b> *(2024)* - Predicting AQI to enhance environmental awareness.</li>
        </ul>
        </div>
        </div>
    """, unsafe_allow_html=True)

    # Contact Section
    st.markdown("""
        <div class='container'>
        <div class='card'>
        <p class='sub-header'>📬 Connect with Me</p>
        <p class='text'>📧 <a href='mailto:prathameshaj2004@gmail.com'>prathameshaj2004@gmail.com</a></p>
        <p class='text'>🌐 <a href='https://www.instagram.com/prathamesh_jadhav_30/' target='_blank'>Instagram</a> | <a href='https://github.com/PrathameshJadhav30' target='_blank'>GitHub</a></p>
        </div>
        </div>
    """, unsafe_allow_html=True)

    st.success("🚀 Always excited to collaborate on innovative projects and bring ideas to life! Let’s connect!")
