NLP Final Task 2026 
RAG-Based Mental Health Support Chatbot 
Introduction 
This project requires designing and implementing a Retrieval-Augmented Generation (RAG) 
based chatbot for a mental health support system. The chatbot must provide grounded, 
empathetic, and context-aware responses to queries related to anxiety, depression, stress, 
and crisis support. 
The system integrates multiple NLP techniques into a single pipeline, where each component 
plays a critical role in improving overall system performance. 
Prerequisites 
Before starting this project, complete the following course: 
• Retrieval Augmented Generation (RAG) @ deeplearning.ai 
ONLY the first 4 modules. 5th module is extra. 
System Overview 
The chatbot system is composed of multiple interconnected modules. Each module 
contributes directly to the performance of the final system, making this a fully integrated, 
end-to-end project rather than isolated tasks. 
Project Modules 
1) Language Detection 
Build a multi-class classifier using Traditional NLP such as: Count Vectorizer or TF-IDF 
Vectorizer. Train ML model to classify the language of the user’s question. This module is very 
important for the entire system for searching right in the knowledgebase in addition to replying in 
the same language and so on. 
2) Emotion Classifier 
Build a multi-class classifier using either Recurrent Neural Networks OR Transformers. Feel 
free to choose as you like. Train model to classify emotion of the user’s question. This module is 
very important for optimizing the final chatbot response depending on the user’s emotion to be 
able to handle his different emotions. This is a key factor in the success of the system. 
3) Intent Classifier 
Build a multi-class classifier using either zero shot or few shot LLM prompting to classify the 
intent of the user’s question. Intent is one of the following:  greeting, goodbye, gratitude, 
asking_mental_health_question, out_of_scope.  This module is very important for routing the 
entire system to the best route for answering. For example: - If the user is greeting, there is no need for RAG and answer directly. - If the user is asking a mental health question, you must use RAG to answer. 
4) Q&A RAG 
Build RAG pipeline using the mental health counseling dataset to answer the upcoming user’s 
questions. Feel free to use the suitable framework you prefer or build from scratch, as you like. 
You need to follow these components: - For vector database, use free cloud qdrant.  - For embeddings, use senetence transformer. - For LLM, use free groq account and gpt-oss-120b or gpt-oss-20b. 
Datasets 
1) Language Identification Dataset 
2) Emotion Dataset 
3) Mental Health Counseling Conversations 
Guidelines 
• You must use python. 
• Choose the most suitable data pre-processing techniques. 
• Use Flask or FastAPI or any suitable web framework to deploy the model locally. 
Deliverables 
• Four module-specific notebooks. 
• Deployment scripts. 
• Any additional files/documentation you need. 
Notes 
• In the assessment phase, you’ll be asked to run your models locally, furthermore, you’ll 
be asked in any technical decision/implementation you’ve made, so be well prepared, 
and avoid overcomplicated approaches you don’t fully grasp. 
• Early submission doesn’t affect your grade, take your time. 