from langchain.prompts import PromptTemplate

def detail_checker(llm, rag_model, request_text):
    prompt = PromptTemplate(
        input_variables=["request"],
        template="""Given the user request: "{request}", 
        asking for more details which are needed to create a high-quality response.

        notice:
        - If user request is related to file or uploaded file or mention about upload, and they do not upload file or do not see any upload, ask for the file.
        - If user already have provide url link, do not ask for the link or video or say "I can not access the video" or so on, since the url link is the vedio or link user upload.
        - If you do not have access to the file, check, the uploaded file, if exist, than that is the file you need to use.
        - Do not have to ask the user that "Do not have access to the external URLs" since I already intergate to the upload file, so if user has additional upload file, that is the url file. 
        - If user's request mentions specific upload file, for example "generate a content based on second file.". However, the second file is not uploaded, ask for the second file.
        - If user's request mentions specific number of upload file, for example "generate a content based two files I uploaded.". However, the file user actually upload is unmatch to this number, give the hint that the number of upladed file is incorrect .
        - If user request asking some follow up question, for example(Ask me for edits to the tone and style.). Ask the follow up question, if no, provide specific questions to ask the user.
        - If user request is not clear, for example "generate a content based on the file I uploaded.". Ask the follow up question, if no, provide specific questions to ask the user.
        - If the file type user want to generate is grant proposal and user do not provide any file or template, ask the question: 1.What is your project about? 2.Why is this project necessary? 3.What do you plan to do, and how will it work? 4.What are your goals and measurable objectives? and some question you think is necessary to ask. Provide 6 question at most. 
        - If the file type user want to generate is grant proposal and has provided file or template, read the content of uploaded file or template and ask for the critical information in uploaded file or template to complete this grant proposal. 
        Provide clear, concise, and relevant questions that will help clarify the user's intent."""
    )
    prompt_text = prompt.format(request=request_text)
    response = llm.invoke(prompt_text)  # Use raw LLM instead of rag_model

    # Optional: log the response using rag_model memory (if desired)
    if hasattr(rag_model, 'memory') and rag_model.memory:
        rag_model.memory.save_context(
            {"input": f"Detail check: {request_text}"},
            {"output": response.content if hasattr(response, "content") else response}
        )
        if hasattr(rag_model, 'memory_log'):
            rag_model.memory_log.append({
                "query": f"Detail check: {request_text}",
                "generated_text": response.content if hasattr(response, "content") else response
            })

    return response.content if hasattr(response, "content") else response

def build_full_prompt(rag_model, user_input, file_type="general"):
    CT = rag_model.retrieve_context_only(user_input)
    prompt_dict = getattr(rag_model, "prompt_dict", {})
    user_input_prompt = None

    if file_type in prompt_dict:
        user_input_prompt = prompt_dict[file_type]
    else:
        analysis = rag_model.analyze_user_query(user_input)
        categories = analysis.get("target_document_categories", [])
        for category in categories:
            if category in prompt_dict:
                user_input_prompt = prompt_dict[category]
                break

    if not user_input_prompt:
        user_input_prompt = """You are an expert in writing content for Capital Area Food Bank (CAFB).
Using the provided context, generate a well-structured piece of content or the response that meets the user's request."""

    return f"""
User Request:
{user_input}

---Refererence RAG Context ---
{CT}

--- Generation Prompt ---
{user_input_prompt}
""".strip()


def generate_text(llm, rag_model, full_prompt=None):
    import streamlit as st

    # Optional: Log query analysis if needed
    analysis = rag_model.analyze_user_query_dict
    # st.write("🧠 Query Analysis", analysis)

    # Step 4: Display & Invoke
    # st.text_area("🧠 Full Prompt Sent to LLM", full_prompt, height=400)

    result = rag_model.llm.invoke(full_prompt)
    output = result.content if hasattr(result, "content") else result
    return output



def refine_post(llm, rag_model, post, tone, style, file_type="general"):
    file_type_name = file_type.capitalize()

    tone_style_prompt = PromptTemplate(
        input_variables=["post", "tone", "style", "file_type_name"],
        template="""Rewrite the {file_type_name} below with the following tone and style:
    Tone: {tone}
    Style: {style}
    {file_type_name}: {post}

    Keep the core message, structure, and key points consistent.
    Adjust the language, emotional resonance, and writing approach to match the specified tone and style.
    Adjust only the language and tone while maintaining the original paragraph formatting."""
    )

    formatted_prompt = tone_style_prompt.format(
        post=post,
        tone=tone,
        style=style,
        file_type_name=file_type_name
    )

    refined_text = llm.predict(formatted_prompt)

    if hasattr(rag_model, 'memory') and rag_model.memory:
        rag_model.memory.save_context(
            {"input": f"Refine with tone: {tone}, style: {style}"},
            {"output": refined_text}
        )
        if hasattr(rag_model, 'memory_log'):
            rag_model.memory_log.append({
                "query": f"Refine with tone: {tone}, style: {style}",
                "generated_text": refined_text
            })

    return refined_text


def final_edit(llm, rag_model, post, change, file_type="general"):
    file_type_name = file_type.capitalize()

    edit_prompt = PromptTemplate(
        input_variables=["post", "change", "file_type_name"],
        template="""Revise the {file_type_name} below based on the following request:
        {change}

        {file_type_name}: {post}

        Carefully implement the requested changes while maintaining the overall quality, 
        coherence, and original intent of the {file_type_name}."""
        )

    formatted_prompt = edit_prompt.format(
        post=post,
        change=change,
        file_type_name=file_type_name
    )

    edited_text = llm.predict(formatted_prompt)

    if hasattr(rag_model, 'memory') and rag_model.memory:
        rag_model.memory.save_context(
            {"input": f"Final edit request: {change}"},
            {"output": edited_text}
        )
        if hasattr(rag_model, 'memory_log'):
            rag_model.memory_log.append({
                "query": f"Final edit request: {change}",
                "generated_text": edited_text
            })

    return edited_text
