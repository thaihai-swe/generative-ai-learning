from openai import OpenAI

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=1000,
    temperature=0.7,
)

input_token_price = 0.15 / 1_000_000
output_token_price = 0.6 / 1_000_000

input_tokens = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens # Output (what the AI wrote)
# Calculate cost
# Cost = (Input Tokens * Input Price) + (Output Tokens * Output Price)
cost = (input_tokens * input_token_price + output_tokens * output_token_price)
response.choices[0].message.content
print(f"Estimated cost: ${cost}")



# Create a request to the Chat Completions endpoint
response = client.chat.completions.create(
  model="gpt-4o-mini",
  max_completion_tokens=150,
    messages=[
        {"role": "system", "content": "You are a helpful Geography tutor that generates concise summaries for different countries."},

        # Example Question (User)
        {"role": "user", "content": "Give me a quick summary of Portugal."},

        # Example Answer (Assistant)
        {"role": "assistant", "content": "Portugal is a country in Europe that borders Spain. The capital city is Lisboa."},

        {"role": "user", "content": "Give me a quick summary of Greece."}
    ]
)

# Extract the assistant's text response
print(response.choices[0].message.content)


client = OpenAI(api_key="<OPENAI_API_TOKEN>")

messages = [{"role": "system", "content": "You are a helpful math tutor that speaks concisely."}]
user_msgs = ["Explain what pi is.", "Summarize this in two bullet points."]

# Loop over the user questions
for q in user_msgs:
    print("User: ", q)

    # Create a dictionary for the user message from q and append to messages
    user_dict = {"role": "user", "content": q}
    messages.append(user_dict)

    # Create the API request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = messages,
        max_completion_tokens=100
    )

    # Append the assistant's message to messages
    assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}
    messages.append(assistant_dict)
    print("Assistant: ", response.choices[0].message.content, "\n")
