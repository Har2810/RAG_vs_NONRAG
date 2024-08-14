#WITHOUT RAG FOR 7.8KB FILE CONTENT


import re
from transformers import AutoTokenizer, AutoModel
import torch
# import chromadb
# import chromadb.utils.embedding_functions as embedding_functions
# from odf.opendocument import load
# from odf.text import P
import google.generativeai as genai
# import os
# from sklearn.metrics.pairwise import cosine_similarity


# huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
#     api_key="hugging_face_api_key",
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # Set up the persistent directory for ChromaDB
# persist_directory = "./chroma_db"
# if not os.path.exists(persist_directory):
#     os.makedirs(persist_directory)

# chroma_client = chromadb.PersistentClient(path=persist_directory)
# collection = chroma_client.get_or_create_collection(name="my_collection")

# file_path = 'Path_to_actual_file'


GOOGLE_API_KEY = "google_ai_studio_api_key"
genai.configure(api_key=GOOGLE_API_KEY)

# def create_chunks(text, chunk_size=200, overlap=50):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start += (chunk_size - overlap)
#     return chunks

# def process_document(file_path):
#     print("Processing document...")
#     doc = load(file_path)
#     content = []
#     for paragraph in doc.getElementsByType(P):
#         paragraph_text = ''.join(node.data for node in paragraph.childNodes if node.nodeType == 3)
#         content.append(paragraph_text.strip())
#     document_text = ' '.join(content)
    
#     chunks = create_chunks(document_text)
#     print(f"Number of chunks created: {len(chunks)}")
    
#     embeddings = huggingface_ef(chunks)
#     print("Generated embeddings for all chunks.")
    
#     ids = [f"chunk_{i}" for i in range(len(chunks))]
#     metadatas = [{"text": chunk} for chunk in chunks]
    
#     collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
#     print("Chunks and embeddings added to Chroma DB.")




# def query_collection(query_text, n_results=10):
#     query_embedding = huggingface_ef([query_text])[0]
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=n_results,
#         include=["metadatas", "embeddings"]
#     )
#     relevant_chunks = [metadata['text'] for metadata in results['metadatas'][0]]
#     chunk_embeddings = results['embeddings'][0]
#     similarity_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
#     return relevant_chunks, similarity_scores.tolist()

def generate_answer(query):
    # if not contexts or max(similarity_scores) <= 0.1:
    #     return "Information not found in the document."

    content = """
    This year the Open Source Festival on March 29, 2014. The day before, on March 28 we will hold a pre-festival day of workshops. Currently, we are inviting people to for presentations. The keynote will be by members of the OpenStack community! And, we anticipate sessions inline with the Maker movement and open hardware, focusing on: Raspberry Pis, Arduinos, 3D printing.
Many open source-related activities took place during 2013 at SUNY Albany, so we're seeing the participation and nature of topics and discussions increase. Some of those activites were: workshops on the use of Git, training sessions on how to write , and training on how to participate in the .
We got involved with the rare diseases community when we were approached by who leads a at Albany Medical Center. Ed was looking for students to help him create an information system for supporting the families in their communities. Given our previous experience at workshops for ASIST, and given the great strides and learning that had been coming from the Open Source Festival, we understood the basics of what was needed to get students and faculty involved and helping. A hackathon was the perfect informal, collaborative environment; and , to work on developing open source software for the rare diseases communities.
What we learned with these hackathon events was that it is very important to create a space where everybody can participate and contribute. This implies that we have to resist the temptation for efficiency (which is a trace of managing scarcity) and instead ensure that activities are engaging and are covered in a redundant manner by multiple people (which is a trace of managing abundance).
It has been very rewarding to see all these volunteers, students, and faculty come together—attending and helping when their schedules allow, and working together for a common good.
By Jodi Biddle (originally published February 2014)
I'm a newcomer to the tech industry. I don't have a degree in Computer Science or Engineering. I'm a writer by trade and training, so coming to work for Red Hat after years of freelancing and crappy office jobs was a real shock. Which is to say, a pleasant shock. Tattoos? Sure. Pink hair? Oh, yes. Start time? Whatever suits you best. And unlike other places I've worked, not a single man has expected me to make them a cup of coffee, and nobody tells me to "smile love, nobody likes a sadsack in the office!" (I frown when I concentrate. I'm sorry! And by that I mean I'm totally not sorry.)
What's more, I work in a department full of women. This was also unexpected—I'd been led to believe that women didn't work in IT, and so I came in bracing for more male-dominated office life. But there are heaps of women in my office. Well, I should qualify—heaps of women on my floor. Engineering? Not so much. There are lots more women in the "periphery" of tech, such as technical writing and business management, but engineering is still hopelessly male-dominated. Why is that?
From the :
"In 2008, women received 57% of all undergraduate degrees but represented only 18% of all Computer and Information Sciences undergraduate degrees. There has been a 79% decline, between 2000 and 2008, in the number of incoming undergraduate women interested in majoring in Computer Science. As a result, only 27% of computer scientists today are female."
I find that really interesting. In the course of researching this piece, I read a lot of troubling stories from women in the industry. This quote from stood out to me:
"It's tiring always being first, always being different, always being the one who has to adapt, denying important parts of yourself just to get the chance to do your job. It’s like being a stranger in a strange land, where you speak the language but nobody learns yours. That's why even women who do well in development end up leaving mid-career."
What is the strange land? What is the language of the tech industry? As a newcomer, and a woman, it's immediate and noticeable, but oddly hard to articulate. It goes deeper than the plethora of buzz-words and over-determined jargon. I think it's a cultural problem, and I don't just mean in the standard "men-and-women-can't-get-along" kind of way.
I don't mean to start a fight here, but I think the tech industry, and open source in particular, is snobby. Geek culture is so deeply insinuated into every part of this industry it forms a barrier of entry to everybody already not inducted by a nearly life-long process of immersion. Just like any culture, there are acceptable and unacceptable ways to dress, shows to watch, books to read, hobbies to engage in, and modes of communication.
It's also understandable to a degree how suspicious geeks can be of non-geeks. The exclusionary nature of geek culture works both ways: mainstream society's relationship with geek culture seems to be torn between opportunistic profiteering (Big Bang Theory) and downright antagonism (every "nerd" movie trope). But an unfortunate by-product of this is a seriously insular culture that has been wrapped around the tech industry. That culture could do a lot to be more welcoming in general, but more welcoming to women specifically.
It returns to this notion of working somewhere "where you speak the language but nobody learns yours." I'm nerdy, geeky, dorky, whatever. I was the weird kid in high school. I read comic books, I play Dungeons and Dragons. I also like shoes, and handbags, and musical theatre, and I pay too much for haircuts. I bulk-bought makeup in Walmart last time I was in the States like there was an oncoming fashion apocalypse.
Despite the fact that I definitely consider myself die-hard geeky, apparently I don't fit the girl-geek stereotype—maybe geek girls don't shave their heads and get a lot of tattoos? I lack the pre-requisite shyness? (It was a long slog but I grew out of it eventually...) I'm actually not sure why I get disqualified. I own a NASA shirt! Nonetheless, I therefore run the risk of being accused of the ultimate insult against geek puritanism: the fake geek girl. I won't speak too much about this recent phenomenon, but apparently now there are girls invading the land of geekdom and appropriating cultural icons as fashion accessories without any knowledge or understanding of their history or significance. I can see how this is problematic, and there's a big debate to be had about cultural appropriation there, but automatically dismissing anyone who isn't deemed "appropriately geeky"—especially women, because this vitriol seems to be focused on women—isn't exactly going to facilitate the kind of growth most of us would like to see in open source. Authenticity should only be discussed in terms of desire to be a part of the open source community, not in terms of what clothes you wear or what books you read.
Not only do I have to constantly prove my worth as a non-technical person in a highly technical world, I also have to contend with the notion that I'm just trying to disguise myself as someone nerdy in order to fit in with the misfits. (How does that logic even work?) So I ask you, if we can't even trust women from our own highly insular culture, then how are other women ever going to feel welcome in our industry, so caught up in said culture? By failing to disrupt the narrative that entwines geek culture with IT, we're alienating everyone who already feels like IT is inaccessible, especially women.
I nominate open source particularly because even by tech industry standards, this is the hardcore stuff. When I tell other IT people I work for a Linux company even they sometimes get the haunted look of somebody about to be bombarded with a bunch of stuff they don't know or care about. I really like it though. I like the passion people have. I like that innovation and progress are the big markers of success, and that good ideas are going to naturally work their way to the top. This sort of natural selection shouldn't be limited by who can produce some geek cred. I know it's worn like a badge of pride, but it shouldn't devalue other social structures, especially feminine social structures, in the process.
It's got to be hard to have the best ideas when we're missing a huge chunk of the population. Imagine how many brilliant potential software engineers are being lost to other industries because they feel like there isn't any space for pink high heels in IT? Should femininity be a foreign language barrier that women need to overcome in order to have a career? And why should women have to prove their geekiness even when it is genuine? We need to better delineate between the tech industry and geek culture, because ultimately, being snobby in open source is bad for business.
By Lauren Egts, Interview with (originally published January 2014)
When you walk into the cavernous, old tire plant of in Akron, Ohio, the last thing that you'd expect to find in this big building is such a "tiny" treasure. Unexpected though it may be, this is where Ken Burns and the TinyCircuits team has set up shop, and it's where they make tiny open source hardware treasures: miniaturized Arduino compatible circuits.
Ken Burns is the founder of TinyCircuits and has always been fascinated with computers. He first got access to a computer, an Apple 2, when he was six years old at a local library, for only 15 minutes a week. He continued working with computers, earned a degree in electrical engineering at the University of Akron, and eventually began working at AVID Technologies, Inc., a company that does product design in Twinsburg.
At AVID, he says, "I noticed a common thread—smart sensor modules people would want put into their products." Eventually, he had the inspiration for and started a business around open source hardware and electronics.
The won them the for Open Source Project of the Year in 2012.
TinyCircuits makes products exactly like what their name implies. The TinyDuino is the size of a quarter but is as powerful as the Arduino Uno. There are many different types of low cost, open source boards that are made at TinyCircuits. The most popular product is the TinyDuino starter kit (includes the processor board, USB board, and proto boards). TinyCircuits' electronics also have the expandability of an Arduino board. Using TinyShields, you can snap on capabilities like an accelerometer, WiFi, and GPS.
"The difference is the miniaturization of it while maintaining the expandability of the core Arduino platform. We have a miniature platform that you can still expand on. It is fairly easy to add WiFi, GPS, motor control. It keeps the simplicity of the Arduino and keeps it useful for projects that need to be small."
TinyCircuits got its start in 2011 with a lot of help from the open source community. The team was able to raise over $100,000 in donations from their Kickstarter campaign to start manufacturing and distributing TinyCircuits. This also kickstarted a community of followers, users, and supporters, particularly those who were already users of Arduino boards. Because TinyCircuit products are compatible with the Arduino products, the company fulfills the need of a niche market and leverages the Arduino community. 
"The traditional view of open source is about software. Open source hardware has been around for about 7 to 10 years. Making hardware open and building a community around it is a huge advantage in hardware like in software," Burns said. "The community behind it keeps it alive, keeps it useful."
Many users have come up with exceptional uses for TinyCircuits due to their low power usage, customability, and size. For example, a PhD student from England was better able to observe climate data over large expanses of land. He used his TinyCircuits to measure the basic environmental factors of an African game preserve for a year, and thus he had more data points and all for a much lower cost than the typical, expensive equipment.
TinyCircuits can also "stir a kid's imagination and make them use their engineering mind even though they might not think they have one yet," said Burns. Users tend to primarily be hobbyists and students who are already somewhat familiar with the Arduino board and other open source electronics, but who have a need to 'miniaturize' their projects.
The Open Voices eook series highlights ways open source tools and open source values can change the world. Read more at .
"""
    prompt = f"""Based on the following context,
,provide a comprehensive answer to the question. If the information is not explicitly mentioned, make reasonable inferences and 
clearly state that you are doing so. If the answer cannot be derived from the given context, 
state that the information is not found in the document.

Context:
{content}

Question: {query}

Answer:"""

    try:
        response = genai.GenerativeModel('gemini-pro').generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.8,
            top_k=40,
            max_output_tokens=250
        ))
        if response.parts:
            return response.text
        else:
            return "Error: The model did not generate any content."
    except Exception as e:
        return f"Error: Unable to generate response. {str(e)}"

# def expand_query(query):
#     prompt = f"Given the question: '{query}', generate 2-3 related questions that might help provide a more comprehensive answer. Format the output as a comma-separated list."
#     try:
#         response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
#         if response.parts:
#             expanded_queries = [query.strip() for query in response.text.split(',')]
#             return [query] + expanded_queries
#         else:
#             return [query]
#     except Exception as e:
#         print(f"Error in query expansion: {str(e)}")
#         return [query]

def main():
    # if collection.count() == 0:
    #     process_document(file_path)
    # else:
    #     print("Document already processed. Skipping embedding creation.")
    
    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        # expanded_queries = expand_query(query)
        # all_chunks = []
        # all_scores = []
        # for eq in expanded_queries:
        #     chunks, scores = query_collection(eq, n_results=5)
        #     all_chunks.extend(chunks)
        #     all_scores.extend(scores)
        answer = generate_answer(query)
        print("\nGenerated Answer:")
        print(answer)

if __name__ == "__main__":
    main()