import streamlit as st

st.set_page_config(
    page_title="Personal Blog Chatbot",
    page_icon="🐳",
)

st.markdown("""

### **Demo Questions Page** 💬
### 💬 Try These Demo Questions!

Not sure what to ask? Here are some sample questions to get you started! 👇

These questions showcase how I can retrieve, rank, and generate answers using multiple document sources. Feel free to modify them or create your own!

- *What are volumes in Kubernetes?*
- *Explain JSON and BSON differences in depth*
- *How to debug kubernetes containers?*
- *What are statefulsets and volumes? Give code for same*


---

### **How to Use These Questions**:
1. Simply copy and paste one of the demo questions into the chatbox 💬.
2. Watch as the bot retrieve documents, re-rank them, and provide a detailed, sourced answer 📄.
3. Feel free to ask follow-up questions or explore new topics!

---
""")

