import webbrowser
import streamlit as st
from streamlit.components.v1 import html


def market_content():
    st.markdown("""
    <div style="background-color: #323232; padding: 20px; border-radius: 15px;">
        <h3 style="text-align: center; margin-top: 0; color: white;">Job Market</h3>
        <p style="font-size:18px; text-align:justify; color: white;"> The job market has evolved significantly over the past decade, influenced by various factors such as technological advancements, economic shifts, and societal changes. 
        In recent years, there has been a growing demand for workers with skills in technology, data science, artificial intelligence, and cybersecurity. </p>
        <p style="font-size:18px; text-align:justify; color: white;"> These fields are expected to continue expanding, driven by innovation and the increasing reliance on digital solutions across industries. However, the rise of automation and artificial intelligence also poses challenges for certain job categories, 
        with routine and manual tasks being replaced by machines.</p>
        <p style="font-size:18px; text-align:justify; color: white;"> As a result, workers in sectors like manufacturing and administrative roles are being displaced, prompting the need for re-skilling and upskilling initiatives to help them transition into new careers. 
        Remote work has become a major trend, accelerated by the COVID-19 pandemic, offering flexibility and a wider talent pool for employers. However, this shift has also led to the need for improved digital infrastructure and collaboration tools to support distributed teams effectively. 
        </p>
        <p style="font-size:18px; text-align:justify; color: white;"> Overall, the job market is increasingly competitive, with employers seeking candidates who possess not only technical expertise but also strong soft skills such as 
        adaptability, communication, and problem-solving. As the global economy continues to change, workers must continuously evolve to stay relevant in an ever-changing job market. </p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("")


def education_content():
    st.markdown("""
    <div style="background-color: #323232; padding: 20px; border-radius: 15px;">
        <h3 style="text-align: center; margin-top: 0; color: white;">Education and Its Role in Shaping the Job Market</h3>
        <p style="font-size:18px; text-align:justify; color: white;"> 
        Education has long been a cornerstone in shaping the career paths of individuals in the modern job market. The demand for educated workers continues to grow as technology advances, industries evolve, and the global economy becomes increasingly interconnected. Higher education, such as a college or university degree, is still seen as a prerequisite for many well-paying jobs, especially in fields like technology, finance, healthcare, and engineering. However, the traditional model of education is facing challenges. The cost of tuition is rising, student debt is a growing concern, and some argue that the education system is not adequately preparing students for the practical challenges of the workplace.
        As the world changes, many employers are looking beyond traditional degrees and instead focusing on skills and experience. This shift is particularly evident in industries like tech, where coding boot camps and certifications have gained popularity. The rise of online learning platforms and remote work opportunities has further altered the educational landscape. Job seekers now have the opportunity to gain relevant skills without the need for a traditional four-year degree. However, despite the growing emphasis on skills, the reality remains that education continues to be a key factor in shaping the job market and determining an individual's success in their career.
        </p>
    </div> """, unsafe_allow_html=True)
    st.subheader("")


# Function to create the second paragraph about salary and its importance
def salary_content():
    st.markdown("""
    <div style="background-color: #323232; padding: 20px; border-radius: 15px;">
        <h3 style="text-align: center; margin-top: 0; color: white;">Salary Trends and the Role of Compensation in Job Selection</h3>
        <p style="font-size:18px; text-align:justify; color: white;"> 
        In the competitive job market, salary remains one of the most important factors for job seekers. It is often the primary motivator for individuals looking to switch jobs, negotiate raises, or pursue higher-paying careers. Salary trends have been shifting over the years, with inflation, the cost of living, and changes in demand for particular job roles influencing compensation packages. In recent years, there has been a noticeable shift towards offering more than just a base salary; employers are increasingly focusing on comprehensive compensation packages that include benefits such as health insurance, retirement savings, paid time off, bonuses, and equity in the company. This shift in approach reflects a more holistic view of employee compensation.
        The technology and finance sectors are known for offering some of the highest salaries, while industries like education and non-profits tend to offer lower compensation. However, salary alone isn't always the deciding factor for job seekers. Work-life balance, career growth opportunities, and company culture are also becoming significant factors in the decision-making process. As the job market evolves, it is essential for job seekers to consider both the financial and non-financial aspects of a job before making career decisions. Understanding salary trends and how they fit into the larger picture of job satisfaction is critical for anyone entering or navigating the workforce.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("")


# Function to add images (You can replace these with your own image paths)
def add_images(image_number):
    if image_number == 1:
        st.image("1.png", caption="Job Market Trends", use_container_width=True)
    elif image_number == 2:
        st.image("2.jpeg", caption="Salary & Benefits", use_container_width=True)
    elif image_number == 3:
        st.image("3.png", caption="Education & Career", use_container_width=True)


# Function to create a sidebar for navigation with buttons to link different Streamlit apps
def create_sidebar():
    st.sidebar.header("Explore More")
    st.sidebar.markdown("""
    <div style="
        background-color: #ED4245; 
        padding: 15px; 
        text-align: left; 
        color: black;
        margin-bottom: 2px; 
        font-size: 18px;">
        <p><em> Conducted an in-depth analysis of occupations across various sectors, highlighting salary changes pre- and post-COVID. </em></p>
    </div>""", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="
        background-color: #FEE75C; 
        padding: 15px; 
        text-align: left; 
        color: black;
        margin-bottom: 2px; 
        font-size: 18px;">
        <p><em> Developed a career recommendation system using content-based filtering to help individuals identify optimal education levels 
        aligned with desired occupations and corresponding pay. </em></p>
    </div>""", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="
        background-color: #5865F2; 
        padding: 15px; 
        text-align: left; 
        color: black;
        margin-bottom: 2px; 
        font-size: 18px;">
        <p><em> Leveraged machine learning to project occupational trends for the 
        next 5 years, providing actionable insights for future career planning.</em> </p>
    </div>""", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="
        background-color: #57F287; 
        padding: 15px; 
        text-align: left; 
        color: black;
        margin-bottom: 2px; 
        font-size: 18px;">
        <p><em>If you want to seamlessly blend into the evolving job market, explore these insights and choose the education and career path that best suits your aspirations.</em></p>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    st.sidebar.markdown("""
    <p><b><em>Click on the buttons below to explore more information on the respective topics.</em></b></p>
    """, unsafe_allow_html=True)

    # First two buttons: For other content inside this app
    if st.sidebar.button("Occupational Pay Analysis"):
        webbrowser.open_new_tab('http://localhost:8502')
        st.success("The other app has been opened in your browser.")

    if st.sidebar.button("Education-Based Recommendations"):
        webbrowser.open_new_tab('http://localhost:8503')
        st.success("The other app has been opened in your browser.")

    # Third button: This links to a different Streamlit app (e.g., app2.py)
    if st.sidebar.button("Future Occupational Projections"):
        webbrowser.open_new_tab('http://localhost:8503')
        st.success("The other app has been opened in your browser.")


# Main function to display content and buttons
def main():
    # Set the page layout to wide mode to make use of the full width
    st.set_page_config(layout="wide")
    st.markdown("""<h1 style="text-align: center; margin-top: 0;">The Job Market: Trends and Insights</h1>""", unsafe_allow_html=True)
    # Add content and images to the page
    add_images(1)  # Display the images first (You can adjust the placement of images as needed)
    # Full width, left-aligned paragraphs
    st.markdown("<div style='text-align: left;'>", unsafe_allow_html=True)
    market_content()
    add_images(3)
    education_content()
    add_images(2)
    salary_content()

    # Create the sidebar buttons linking to other Streamlit apps (you can replace with URLs)
    create_sidebar()

    st.markdown("""<p style="font-size:30px; text-align:center;"><b><em> To explore detailed pages on each topic, click the buttons in the sidebar.</em></b></p>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()