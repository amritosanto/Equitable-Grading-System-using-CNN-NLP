import h5py
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

demo_answers = [
    ("ml? no idea. i'm not familiar with that term, sorry. it might be an abbreviation or acronym, but i can't say for sure.", 0),
    ("ml? i'm not sure what that stands for. it's not a term i'm familiar with, so i can't provide any insights on it.", 0),
    ("ml? could be a typo or abbreviation. i'm not familiar with the term, so i can't offer any explanation.", 0),
    ("ml? i'm not familiar with that acronym. it doesn't ring a bell, sorry.", 0),
    ("ml? i've heard the term before, but i'm not entirely sure what it means. it might be related to technology or machines, but i can't say for certain.", 0),
    ("ml, possibly stands for machine learning. i vaguely recall hearing it mentioned in tech discussions, but i'm not entirely sure.", 1),
    ("ml? it might be related to computers or data analysis, but i'm not entirely certain. i've heard the term before, but i can't provide a definitive explanation.", 1),
    ("ml: could be related to algorithms and data processing. i'm not entirely sure, but it sounds familiar.", 1),
    ("ml, possibly an abbreviation for machine learning. i've heard it mentioned in discussions about ai, but i can't recall the specifics.", 1),
    ("ml? it sounds familiar, but i can't remember what it stands for. it might be related to technology or computing, but i'm not entirely sure.", 1),
    ("ml, possibly an abbreviation for something technical. i've heard the term before, but i'm not entirely sure what it means.", 2),
    ("ml? it might be related to computers or programming. i'm not entirely certain, but it sounds familiar.", 2),
    ("ml, possibly involves machines and technology. i've heard the term before, but i can't provide a definitive explanation.", 2),
    ("ml? it sounds like it could be a technical term, but i'm not entirely sure. i'm familiar with the acronym, but i can't recall the specifics.", 2),
    ("ml, possibly related to technology or computing. i've heard the term before, but i can't recall the specifics.", 2),
    ("ml, possibly stands for machine learning. i've heard it mentioned in discussions about ai and data analysis, but i'm not entirely sure.", 3),
    ("ml? it might be related to artificial intelligence or data science, but i'm not entirely certain. i'm familiar with the term, but i can't provide a definitive explanation.", 3),
    ("ml, heard of it in the context of technology and computing. i'm not entirely sure, but it sounds familiar.", 3),
    ("ml? i've heard the term before, but i'm not entirely clear on what it means. it might be related to algorithms and data processing, but i can't say for certain.", 3),
    ("ml, possibly related to computers or programming. i'm not entirely certain, but it sounds familiar.", 3),
    ("ml stands for machine learning, a subset of ai involving data analysis and algorithms. it's used to teach computers to learn from data and make decisions autonomously.", 4),
    ("ml? it might be related to computers or data analysis. i'm familiar with the term, but i can't provide a definitive explanation.", 4),
    ("ml, heard of it in discussions about technology and computing. i'm not entirely sure, but it sounds familiar.", 4),
    ("ml? it sounds like it could be related to algorithms and data processing. i'm familiar with the term, but i can't recall the specifics.", 4),
    ("ml, possibly related to technology or computing. i'm not entirely certain, but it sounds familiar.", 4),
    ("ml encompasses techniques enabling systems to learn from data and make predictions autonomously. it's used in various fields like ai and data science for automated decision-making.", 5),
    ("ml involves algorithms that improve with data exposure, enabling predictive analysis in fields like finance and healthcare.", 5),
    ("ml? it might be related to artificial intelligence or data science. i'm familiar with the term, but i can't provide a definitive explanation.", 5),
    ("ml, heard of it in discussions about technology and computing. i'm not entirely sure, but it sounds familiar.", 5),
    ("ml? it sounds like it could be a technical term, but i'm not entirely sure. i'm familiar with the acronym, but i can't recall the specifics.", 5),
    ("ml employs statistical techniques for computers to learn from data and make predictions autonomously. it's used in various applications like recommendation systems and fraud detection.", 6),
    ("ml algorithms adjust parameters based on data feedback, continuously improving performance. they're used in fields like finance and marketing for predictive modeling.", 6),
    ("ml enables computers to recognize patterns and make data-driven decisions without human intervention. it's used in applications like autonomous vehicles and medical diagnostics.", 6),
    ("ml involves training algorithms to improve performance on tasks through experience and data exposure. it's used in fields like natural language processing and image recognition.", 6),
    ("ml algorithms discern patterns in data to make predictions, adapting and improving over time. they're used in applications like customer churn prediction and stock market analysis.", 6),
    ("ml leverages algorithms to recognize patterns and make data-driven decisions, enhancing automation in various industries.", 7),
    ("ml involves the development of algorithms that improve performance through experience and data analysis. it's used in fields like cybersecurity and supply chain management.", 7),
    ("ml enables computers to process and interpret data for tasks such as classification and prediction. it's used in applications like credit scoring and medical imaging.", 7),
    ("ml algorithms, such as neural networks, analyze data to make predictions or decisions autonomously. they're used in applications like speech recognition and sentiment analysis.", 7),
    ("ml empowers systems to learn from data and adapt to new situations, driving innovation in ai and robotics.", 7),
    ("ml encompasses a range of techniques enabling systems to learn from data and improve performance across various domains.", 8),
    ("ml algorithms, from simple regression to complex deep learning models, enable data-driven decision-making in fields like e-commerce and manufacturing.", 8),
    ("ml enables computers to understand and interpret complex data, revolutionizing industries like healthcare and finance.", 8),
    ("ml techniques, including supervised and unsupervised learning, enable computers to extract insights from data for applications like anomaly detection and customer segmentation.", 8),
    ("ml algorithms adapt and improve autonomously through experience, driving advancements in ai and machine learning research.", 8),
    ("ml algorithms, such as deep learning and reinforcement learning, enable computers to learn from data and optimize performance in applications like autonomous driving and medical diagnosis.", 9),
    ("ml powers innovations from autonomous vehicles to personalized medicine, driving technological advancements in fields like robotics and healthcare.", 9),
    ("ml algorithms process data to identify patterns and make predictions, enhancing decision-making in diverse domains like retail and telecommunications.", 9),
    ("ml enables systems to automate tasks and make informed decisions, revolutionizing industries like manufacturing and energy.", 9),
    ("ml techniques, such as neural networks and decision trees, empower computers to make predictions based on data patterns, enabling applications like predictive maintenance and fraud detection.", 9),
    ("machine learning encompasses techniques allowing systems to learn from data, identify patterns, and make decisions autonomously, powering innovations from autonomous vehicles to personalized medicine.", 10),
    ("ml algorithms enable computers to learn from data and improve performance on tasks, driving innovation and progress in ai and machine learning research.", 10),
    ("ml is foundational to modern ai, powering innovations from autonomous vehicles to personalized medicine, revolutionizing industries and driving technological advancements.", 10),
    ("ml algorithms learn from data to make predictions or decisions without explicit programming, revolutionizing industries and driving technological advancements.", 10),
    ("machine learning enables systems to automatically improve from experience, making it foundational to various applications like recommendation systems and fraud detection.", 10),
    ("ml? no idea. i'm not familiar with that term, sorry. it might be an abbreviation or acronym, but i can't say for sure.", 0),
    ("ml? i'm not sure what that stands for. it's not a term i'm familiar with, so i can't provide any insights on it.", 0),
    ("ml could be a typo or abbreviation. i haven't come across that term before, so i can't offer any explanation.", 0),
    ("ml escapes me right now. is it an abbreviation you're referring to?", 0),
    ("ml? that's a new one on me. it might be a technical term i'm not familiar with.", 0),
    ("ml stands for machine learning. think of it as training a computer to learn from data, like photos or emails. this helps it improve at tasks like recognizing faces or filtering spam.", 6),
    ("ml? it's like a computer program that gets better by itself! it analyzes data to learn and make decisions, without needing explicit instructions.", 6),
    ("ever wondered how social media recommends things you might like? that's ml! it analyzes data to find patterns and make predictions.", 6),
    ("there are different ways ml works. supervised learning involves giving the computer labeled data, like showing it spam and non-spam emails. unsupervised learning lets the computer discover patterns on its own.", 7),
    ("ml powers many things you use daily! from fraud detection to personalized news feeds, ml analyzes vast amounts of data to improve experiences and security.", 7),
    ("ml algorithms are like detectives, finding hidden patterns in data. this helps them make predictions on new data, like recommending movies you might enjoy.", 7),
    ("self-driving cars use ml to navigate! by analyzing data from cameras and sensors, the car 'learns' to identify objects and make safe driving decisions.", 8),
    ("have you ever spoken to a virtual assistant? ml helps them understand your voice commands and respond in a helpful way.", 8),
    ("ml is revolutionizing healthcare. for example, it can analyze medical images to detect diseases like cancer earlier.", 8),
    ("ml algorithms are inspired by the human brain! neural networks, a type of ml, use interconnected nodes that process information similarly to our neurons.", 9),
    ("the more data you feed an ml model, the better it performs. this is why large tech companies with massive datasets are leaders in ml development.", 9),
    ("deep learning, a branch of ml, uses complex artificial neural networks to tackle intricate problems like image and speech recognition.", 10),
    ("machine learning ethics is crucial. it focuses on ensuring fairness, accountability, and transparency in how ml models are developed and used.", 10),
    ("ml isn't perfect. it can be biased based on the data it's trained on. that's why data quality and diversity are important.", 8),
    ("while ml can do amazing things, it can't replace human intelligence (yet!). it's a powerful tool that complements our abilities.", 7),
    ("ml is used in finance to detect fraudulent transactions. by analyzing spending patterns, it can identify suspicious activities that might indicate fraud.", 8),
    ("ml helps create smart homes that adjust to your preferences. for example, a thermostat can learn your ideal temperature and adjust automatically.", 8),
    ("ml is used in weather forecasting to analyze vast amounts of data and make more accurate predictions.", 8),
    ("ml can personalize your learning experience. educational platforms can use ml to tailor learning materials to your strengths and weaknesses.", 8),
    ("ml has the potential to revolutionize many industries, from healthcare to manufacturing. it's an exciting field with vast possibilities.", 8),
    ("as ml continues to develop, it's important to consider the ethical implications. we need to ensure that ml is used for good and benefits everyone.", 8),
    ("imagine a chef who gets better at cooking certain dishes the more they practice. that's similar to how an ml model improves with more data.", 7),
    ("think of ml as a detective sifting through clues (data) to solve a mystery (make a prediction).", 7),
    ("ml is like a language model that learns from the vast amount of text it's exposed to. that's how i can communicate and generate human-like text!", 7),
    ("some people worry that ml will take over the world. relax, robots are still struggling with folding laundry... for now.", 6),
    ("netflix uses ml to recommend shows you might enjoy based on your viewing history. it's like having a personal entertainment assistant!", 7),
    ("ever wondered how ride-sharing apps estimate fares? ml analyzes factors like distance, traffic, and demand to provide accurate estimates.", 7),
    ("ml is used in social media to filter out inappropriate content. it helps create a safer and more positive online environment.", 7),
    ("ml models can be biased if the data they're trained on is biased. it's important to ensure data quality and diversity to avoid unfair outcomes.", 8),
    ("ml models can be computationally expensive to train and run. this can limit their accessibility for smaller organizations.", 8),
    ("while ml can automate many tasks, human expertise is still essential. humans set goals, interpret results, and ensure ethical implementation of ml.", 7),
    ("ml can be a powerful tool to augment human capabilities, not replace them. it allows humans to focus on more complex tasks while ml handles the routine.", 7),
    ("ml is closely linked to artificial intelligence (ai). it's a subfield of ai that focuses on using data to enable machines to learn and improve.", 8),
    ("ml relies on other fields like statistics and computer science to develop algorithms and analyze data effectively.", 8),
]

vocabulary = [
    "machine", "learning", "algorithm", "data", "analysis", "artificial", "intelligence", 
    "neural", "network", "deep", "reinforcement", "supervised", "unsupervised", 
    "model", "regression", "prediction", "classification", "clustering", "anomaly", 
    "recognition", "processing", "feedback", "parameter", "optimization", "decision", 
    "statistical", "technique", "pattern", "training", "experience", 
    "performance", "automation", "development", "interpret", "segmentation", 
    "ethical", "bias", "implementation", "computational", "neuron", "dataset", 
    "artificial", "intelligent", "language", "computation", "statistics", "science", 
]
vocabulary = set(vocabulary)
vocabulary = list(vocabulary)


vectorizer = CountVectorizer(vocabulary=vocabulary)
encoded_answers = []
labels = []
for answer, label in demo_answers:
    encoded_answer = vectorizer.transform([answer]).toarray()[0]
    encoded_answers.append(encoded_answer)
    labels.append(label)


encoded_answers = np.array(encoded_answers)
labels = np.array(labels)


with h5py.File("ml_dataset.h5", "w") as f:
    f.create_dataset("encoded_answers", data=encoded_answers)
    f.create_dataset("labels", data=labels)
    f.create_dataset("vocabulary", data=np.array(vocabulary, dtype="S"))
