# Darshit-LLM-Models

A comprehensive learning repository for Language Models and LangChain concepts.

## 📚 What are Language Models?

Language models are **machine learning models trained to understand, generate, and interact with human languages**. They serve as the backbone of modern digital interactions, making technology more intuitive and natural.

### Core Concept
- **Purpose**: Help machines understand and generate human language
- **Function**: Predict what comes next in a sequence based on learned patterns
- **Example**: After "peanut butter" → predicts "jelly"
- **Output**: Probability of word sequences being valid (human-like, not just grammatically correct)

## 🏗️ Types of Language Models

### 1. Probabilistic Language Models
- **Based on**: N-gram probabilities
- **Method**: Predict next word from preceding n words
- **N-gram Example**: "I love dogs"
  - Bigrams: "I love", "love dogs"
- **Limitation**: Cannot capture deep context

#### 💡 Example:
```
Text: "I love dogs"
Bigrams: ["I love", "love dogs"]
Prediction for "I love ___": Based only on frequency of "I love X" in training data
Limitation: Cannot understand "I love dogs because they are loyal and friendly"
```

### 2. Neural Network-Based Language Models
- **Method**: Use attention mechanisms for contextual understanding
- **Advantage**: Excel at predicting next word with deep context
- **Architecture**: Primarily based on Transformers

#### 💡 Example:
```
Context: "The dog was barking loudly. The neighbor complained about the noise."
Next word prediction considers:
- "dog" relates to "barking" and "noise"
- "neighbor" connects to "complained"
- Full context influences next word choice
```

## 🤖 Transformer Architecture Types

### Encoder-Only
- **Example**: BERT (Bidirectional Encoder Representation Transformer)
- **Use Case**: Understanding and encoding text


#### 💡 BERT Example:
```
Task: Fill in the blank
Input: "The [MASK] was delicious at the restaurant"
BERT considers both left and right context to predict "food", "meal", "pizza"
```

### Decoder-Only
- **Example**: GPT (Generative Pre-trained Transformer)
- **Use Case**: Text generation

#### 💡 GPT Example:
```python
# API call example
prompt = "Write a Python function to calculate fibonacci:"
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
# Generates complete function with explanation
```

### Encoder-Decoder
- **Example**: T5 Model
- **Use Case**: Text-to-text transformations


#### 💡 T5 Example:
```
Input: "translate English to French: Hello, how are you?"
Output: "Bonjour, comment allez-vous?"

Input: "summarize: [long article text]"
Output: "Key points: 1. Main topic... 2. Supporting evidence..."
```

## 🧠 Key Features of Large Language Models (LLMs)

### 1. Emergent Abilities
- Capabilities not observed in smaller models
- **Example**: GPT-3's few-shot learning vs GPT-2's limitations

### 2. Self-Attention Mechanism
- Understands contextual relationships between words in sequences
- Enables deep contextual understanding

### 3. In-Context Learning
- Can learn from examples provided in the prompt
- No additional training required for new tasks


#### 💡 Few-Shot Learning Example:
```
Prompt:
"Translate these phrases:
English: Hello → Spanish: Hola
English: Thank you → Spanish: Gracias
English: Good morning → Spanish:"

Output: "Buenos días"
```

## 💡 Working with LLMs

### Access Methods
- **UI Interfaces**: ChatGPT, Claude, etc.
- **APIs**: GPT-4 API, Anthropic API, etc.
- **Key Skill**: Understanding prompt formatting and LLM behavior

### Development Considerations
- Blends research and engineering
- Requires experience in:
  - Large-scale data processing
  - Distributed parallel training
  - Prompt engineering

## 🎯 Practical Applications

- **Digital Assistants**: Weather queries, general assistance
- **Translation**: Document translation services
- **Search Engines**: Enhanced search capabilities
- **Operating Systems**: AI-integrated OS features
- **Development Tools**: Code completion, documentation

## 🔮 Future Implications

- **Increasing Integration**: LLMs will underpin more technologies
- **Digital Experience**: Reshaping how we interact with technology
- **Critical Evaluation**: Need to understand capabilities and limitations
- **Tool Optimization**: Maximize potential through proper utilization

## ⚙️ How Language Models Generate Text

LLMs are essentially **"text in, text out" machines** that follow a systematic process for text generation.

### High-Level Process
1. Start with an initial sequence
2. Use LLM to generate the next **token**
3. Append token to sequence
4. Repeat until desired sequence is complete

> **Token**: A piece/chunk of text - can be a character, word, or punctuation (parentheses, question marks, etc.)

### The Three-Step Generation Process

#### 1. 🔤 Encoding
- **Input Processing**: Text converted into sequence of vectors
- **Embedding Layer**: Maps each token to vector representation
- **Semantic Capture**: Similar words placed closer in vector space
  - Example: "cat" and "kitten" vectors are positioned close together
- **Context Awareness**: Self-attention mechanism considers word relationships

#### 2. 🎯 Prediction
- **Pattern Recognition**: Uses learned language patterns
- **Probability Distribution**: Softmax layer outputs probabilities for next possible tokens
- **Context-Aware**: Self-attention ensures deeper comprehension
- **Selection**: Model selects most probable next token

#### 3. 📝 Decoding
- **Vector Translation**: Converts vector representation back to human-readable text
- **Sequential Generation**: Output generated one word at a time
- **Contextual Building**: Each new word based on previous sequence

### 🎛️ Control Parameters

#### Temperature
- **Purpose**: Controls model "creativity"
- **Low Temperature (< 1)**: More deterministic, consistent outputs
- **High Temperature (> 1)**: More variability and creativity
- **Zero Temperature**: Nearly identical outputs each time
- **Example**: "The sky is..."
  - Temperature 0: "blue"
  - Higher temperature: "falling" or "filled with stars"


#### 💡 Temperature Examples:
```python
# Low temperature (0.2) - Consistent outputs
prompt = "The capital of France is"
# Output: "Paris" (almost always the same)

# High temperature (1.5) - Creative outputs  
prompt = "Write a story about a cat"
# Outputs vary: "magical adventure", "detective mystery", "space exploration"
```

#### Top-K Sampling
- **Function**: Restricts model to top-k most probable tokens
- **Characteristic**: More rigid approach
- **Use Case**: When you want focused, probable outputs


#### 💡 Top-K Example:
```python
# Top-K = 3
Next word probabilities:
- "the" (0.4)
- "a" (0.3) 
- "an" (0.2)
- "some" (0.1) ← excluded

Model chooses only from top 3 options
```

#### Top-P (Nucleus Sampling)
- **Function**: Selects tokens whose probabilities sum to threshold P
- **Characteristic**: More flexible than top-k
- **Use Case**: Balances creativity with coherence


#### 💡 Top-P Example:
```python
# Top-P = 0.8 (80% cumulative probability)
Word probabilities:
- "happy" (0.5) ✓
- "excited" (0.2) ✓  
- "joyful" (0.1) ✓  ← reaches 80%
- "elated" (0.1) ✗  ← excluded
- "thrilled" (0.1) ✗ ← excluded
```

### 📊 Quality Factors

Text generation quality depends on:
- **Model Size**: Number of parameters
- **Training Data**: Amount and quality of data
- **Training Tokens**: Number of tokens seen during training

### 🎨 Balancing Creativity and Control

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| Temperature | Deterministic, "correct" answers | Creative, varied outputs |
| Top-K | Very focused | Broader token selection |
| Top-P | Conservative choices | More diverse vocabulary |

### Best Practice Guidelines
- **Low Temperature**: Use for factual tasks, Q&A, summarization
- **Higher Temperature**: Use for creative writing, brainstorming
- **Combine Parameters**: Adjust multiple parameters for optimal balance

## 🎓 Training, Fine-tuning, and Learning Methods

Understanding how LLMs are trained and adapted is crucial for working effectively with these models.

### 🏗️ Pre-training
**The foundational phase where models learn basic language understanding**

- **Process**: Feed massive datasets of text to the model
- **Data Sources**: Books, websites, written material
- **Goals**:
  - Recognize language patterns
  - Understand grammar and word usage
  - Learn stylistic elements
- **Output**: Base model with general language understanding
- **Analogy**: Like teaching someone to read and write in a language


#### 💡 Pre-training Example:
```python
# Simplified training data
training_texts = [
    "The sun rises in the east",
    "Python is a programming language", 
    "Machine learning requires data",
    # ... billions more examples
]

# Model learns patterns like:
# "The sun" → usually followed by "rises", "sets", "shines"
# "Python is" → often followed by "a", "used", "popular"
```

### 🎯 Fine-tuning
**Specializing the base model for specific tasks, domains, or behaviors**

- **Process**: Additional training on specialized datasets
- **Purpose**: Adapt general understanding to specific applications
- **Examples**:
  - **Legal Model**: Fine-tune on legal documents
  - **Instruction Model**: Train on instruction-following datasets
  - **Chat Model**: Use conversational datasets with multi-turn dialogues
- **Result**: More accurate and relevant responses in specialized contexts


#### 💡 Fine-tuning Examples:

**Legal Domain:**
```python
legal_dataset = [
    "The defendant shall appear in court on...",
    "According to statute 15.2.3, the plaintiff...",
    "Objection, your honor. Leading the witness..."
]
# Result: Model speaks "legalese"
```

**Code Generation:**
```python
code_dataset = [
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "class Vehicle: def __init__(self, make, model): self.make = make...",
]
# Result: Model generates syntactically correct code
```

### 🧠 In-Context Learning
**Model adaptation based on immediate prompt context (no additional training)**

- **Method**: Leverages existing knowledge and contextual interpretation
- **Key Feature**: No retraining required
- **Capability**: Responds to new, unseen prompts by understanding context
- **Examples**:
  - Few-shot learning from examples in prompt
  - Style adaptation based on prompt format
  - Task understanding from context clues
- **Advantage**: Flexible response to novel requests


#### 💡 In-Context Learning Examples:

**Style Adaptation:**
```
Prompt: "Write like Shakespeare: Modern life is busy"
Output: "Verily, the pace of life in these modern times doth quicken beyond measure"
```

**Task Learning:**
```
Prompt: "Extract emails from text:
Text: Contact John at john@email.com or call 555-0123
Emails: john@email.com

Text: Reach out to support@company.org for help
Emails:"
Output: "support@company.org"
```

### 🔍 Retrieval-Augmented Generation (RAG)
**Hybrid approach combining LLMs with external information retrieval**

#### How RAG Works:
1. **Data Preparation**: Embed dataset into vectors
2. **Storage**: Store vectors in vector database
3. **Query Processing**: Search database for relevant information
4. **Prompt Augmentation**: Insert retrieved info into LLM prompt
5. **Enhanced Response**: Generate answer with external context


#### 💡 RAG Example:

```python
# Step 1: User asks question
user_query = "What are the latest COVID-19 vaccination guidelines?"

# Step 2: Vector search finds relevant documents
retrieved_docs = [
    "CDC guidelines updated March 2024: Booster shots recommended...",
    "WHO recommendations for immunocompromised individuals..."
]

# Step 3: Augmented prompt
augmented_prompt = f"""
Context: {retrieved_docs}
Question: {user_query}
Please provide a comprehensive answer based on the context.
"""

# Step 4: LLM generates informed response
# Output: Accurate, up-to-date information with citations
```

#### 💡 RAG Implementation Flow:
```python
# Simplified RAG pipeline
def rag_pipeline(question):
    # 1. Embed the question
    query_vector = embed(question)
    
    # 2. Search vector database
    relevant_docs = vector_db.search(query_vector, top_k=5)
    
    # 3. Create context
    context = "\n".join([doc.content for doc in relevant_docs])
    
    # 4. Generate response
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = llm.generate(prompt)
    
    return response, relevant_docs
```

#### Benefits:
- **Accuracy**: More precise responses with external data
- **Relevancy**: Context-aware information retrieval
- **Currency**: Access to up-to-date information
- **Specificity**: Handle detailed, specialized queries

#### Use Cases:
- Knowledge bases and documentation
- Real-time information systems
- Domain-specific Q&A systems
- Fact-checking applications

### 🔄 Model Types Based on Training

| Model Type | Training Method | Use Case |
|------------|----------------|----------|
| **Base Models** | Pre-training only | Foundation for further development |
| **Instruction-Tuned** | Pre-training + Instruction fine-tuning | Following specific commands |
| **Chat-Tuned** | Pre-training + Conversational fine-tuning | Multi-turn conversations |

### 🎯 The Training Hierarchy

```
Pre-training (Foundation)
    ↓
Fine-tuning (Specialization)
    ↓
In-Context Learning (Adaptation)
    ↓
RAG (Enhancement)
```

### Key Takeaways
- **Pre-training**: Provides foundational language understanding
- **Fine-tuning**: Adapts understanding to specific domains/styles
- **In-Context Learning**: Applies understanding flexibly to immediate input
- **RAG**: Enriches responses with external, up-to-date information

*Each method represents a unique aspect of LLM development, reflecting the balance between foundational learning, specialized adaptation, contextual responsiveness, and data-enriched generation.*

## 🎨 Prompt Engineering

**The art and science of crafting effective prompts to guide LLMs in generating desired outputs**

### Core Definition
- **Prompt Engineering**: Strategically crafting input prompts to guide LLMs toward specific responses
- **Goal**: Bridge between human intent and machine understanding
- **Key Principle**: Clearer and more specific prompts = better outputs

### 🔄 Prompt Engineering vs In-Context Learning

| Aspect | Prompt Engineering | In-Context Learning |
|--------|-------------------|-------------------|
| **Nature** | Human skill/technique | Model capability |
| **Purpose** | Craft effective instructions | Adapt based on context |
| **Control** | User-driven | Model-driven |
| **Example** | "Write in formal tone..." | Model recognizes formal examples and adapts |

#### 💡 Distinction Example:
```
Prompt Engineering (Human craft):
"You are a professional email writer. Write a formal response to a client complaint about delayed delivery."

In-Context Learning (Model capability):
The model recognizes the formal business context and automatically uses appropriate language, structure, and tone.
```

### 📈 Evolution of Prompt Engineering

#### Early Stage: Simple Q&A
```
Early approach: "What is the capital of France?"
Focus: Direct information extraction
```

#### GPT-3 Era: Advanced Reasoning
```
Modern approach: "Analyze the economic factors that make Paris the ideal capital for France, considering historical, geographical, and political aspects."
Focus: Complex reasoning and structured responses
```

#### Current Era: Multi-Modal & Chain Reasoning
```
Advanced approach: "Using step-by-step reasoning, explain how to solve this coding problem, then provide the implementation with comments explaining each part."
Focus: Structured thinking and comprehensive solutions
```

### 🛠️ Prompt Engineering Techniques

#### 1. Zero-Shot Learning
**Model performs tasks without examples**

```python
# Zero-shot example
prompt = "Classify this email as spam or not spam: 'Congratulations! You've won $1,000,000!'"
# Model uses general knowledge to classify
```

#### 2. One-Shot Learning
**Model learns from single example**

```python
# One-shot example
prompt = """
Example: Email: "Meeting at 2 PM" → Category: Work
Classify: "Doctor appointment tomorrow" → Category:
"""
# Output: "Personal"
```

#### 3. Few-Shot Learning
**Model learns from multiple examples**

```python
# Few-shot example
prompt = """
Sentiment analysis examples:
Text: "I love this product!" → Sentiment: Positive
Text: "This is terrible" → Sentiment: Negative  
Text: "It's okay, nothing special" → Sentiment: Neutral
Text: "Amazing quality and fast delivery!" → Sentiment:
"""
# Output: "Positive"
```

#### 4. Chain-of-Thought Prompting
**Guide model through step-by-step reasoning**

```python
# Chain-of-thought example
prompt = """
Question: A restaurant bill is $85. If you want to tip 18%, how much total will you pay?

Let me think step by step:
1. Calculate the tip: $85 × 0.18 = $15.30
2. Add tip to bill: $85 + $15.30 = $100.30
3. Total amount: $100.30

Question: A shirt costs $45. If there's a 25% discount, what's the final price?

Let me think step by step:
"""
```

### 🎯 Advanced Prompting Strategies

#### Role-Based Prompting
```python
prompt = """
You are a senior software engineer with 10+ years of Python experience. 
Review this code and provide detailed feedback on:
1. Code quality and best practices
2. Performance optimizations
3. Security considerations

[code here]
"""
```

#### Context Stacking
```python
prompt = """
Context 1: You are writing for a technical blog
Context 2: Your audience consists of beginner programmers  
Context 3: The topic is machine learning basics
Context 4: Keep explanations simple but accurate

Task: Explain what a neural network is.
"""
```

#### Constraint-Based Prompting
```python
prompt = """
Write a product description with these constraints:
- Exactly 50 words
- Include keywords: "sustainable", "innovative", "premium"
- Target audience: environmentally conscious consumers
- Tone: Professional but approachable
- Include a call-to-action

Product: Bamboo laptop stand
"""
```

### 🔧 Prompt Optimization Techniques

#### 1. Iterative Refinement
```python
# Version 1 (Basic)
"Summarize this article"

# Version 2 (Improved)
"Summarize this article in 3 bullet points focusing on key findings"

# Version 3 (Optimized)
"Create a concise summary of this article with:
- 3 main findings as bullet points
- Target audience: business executives
- Focus on actionable insights"
```

#### 2. Template-Based Prompting
```python
template = """
Role: {role}
Task: {task}
Context: {context}
Format: {output_format}
Constraints: {constraints}

Input: {user_input}
"""

# Usage
filled_prompt = template.format(
    role="Data analyst",
    task="Analyze trends",
    context="E-commerce sales data",
    output_format="Executive summary with charts",
    constraints="Under 500 words",
    user_input="Q3 sales data..."
)
```

### 🎪 Creative Prompting Techniques

#### Persona Prompting
```python
prompt = """
You are Sherlock Holmes. A user has lost their keys. 
Use your deductive reasoning skills to help them figure out where they might be.
Ask probing questions and provide logical deductions.

User: "I can't find my keys anywhere!"
Sherlock:
"""
```

#### Scenario-Based Prompting
```python
prompt = """
Scenario: You're a startup founder pitching to investors
Situation: 5 minutes left, investor seems skeptical about market size
Your task: Convince them with compelling market data

Deliver your pitch:
"""
```

### ⚠️ Common Pitfalls and Solutions

#### Problem: Vague Instructions
```python
# Bad
"Write something about AI"

# Good  
"Write a 300-word beginner-friendly explanation of how AI chatbots work, including 2 real-world examples"
```

#### Problem: Conflicting Instructions
```python
# Bad
"Be creative but stick to facts. Be brief but comprehensive."

# Good
"Write a factual but engaging 200-word summary that highlights the most important points"
```

#### Problem: Assuming Model Knowledge
```python
# Bad
"Use the latest 2024 regulations"

# Good
"Based on the following 2024 regulations [provide context], analyze this scenario"
```

### 🚀 Integration with External Systems

#### RAG-Enhanced Prompting
```python
prompt = f"""
Based on the following retrieved documents:
{retrieved_context}

Answer the user's question with citations:
Question: {user_question}

Please provide:
1. A direct answer
2. Supporting evidence from the documents
3. Document citations in [1], [2] format
"""
```

### 📊 Measuring Prompt Effectiveness

#### Key Metrics:
- **Accuracy**: Does it produce correct information?
- **Relevance**: Does it address the actual request?
- **Consistency**: Does it produce similar quality across runs?
- **Efficiency**: Does it achieve goals with minimal tokens?

#### A/B Testing Prompts
```python
# Test different approaches
prompt_a = "Summarize this in 100 words"
prompt_b = "Create a concise summary highlighting the 3 most important points"

# Compare outputs for quality, relevance, user satisfaction
```

### 🔮 Future of Prompt Engineering

#### Emerging Trends:
- **Multi-modal prompting**: Text + images + code
- **Dynamic prompting**: Adaptive based on context
- **Collaborative prompting**: Human-AI prompt refinement
- **Domain-specific prompt libraries**: Pre-optimized templates

#### 💡 Best Practices Summary:
1. **Be Specific**: Clear instructions yield better results
2. **Provide Context**: Help the model understand the situation
3. **Use Examples**: Show don't just tell
4. **Iterate and Refine**: Test and improve prompts
5. **Consider the Model**: Tailor to specific LLM capabilities
6. **Structure Clearly**: Use formatting for complex requests

*"Prompt engineering is not just a technical skill, it's an art. It requires a deep understanding of a model's capabilities and your desired outcome."*

## 📖 Learning Path

This repository will document my journey through:
1. **LangChain Fundamentals**
2. **Model Architecture Deep Dives**
3. **Prompt Engineering Techniques**
4. **Custom Model Development**
5. **Real-world Applications**

---

*"As we become increasingly reliant on AI assistance, a foundational grasp of language models equips us to better appreciate this technology's marvel."*

# LangChain Retrieval Guide

## Overview of Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation allows language models to access external data that wasn't part of their training. This is particularly useful for:
- Company-specific documentation
- Recent or private information
- Domain-specific knowledge

## Core Components & Flow

1. External Data Sources → Load Data → Split into Chunks → Create Vector Embeddings → Store in Vector Store → Retrieve Relevant Documents

## Document Loaders

Document loaders transform various data sources into LangChain document objects.

```javascript
import { TextLoader } from 'langchain/document_loaders/fs/text';

const loader = new TextLoader('path/to/file.txt');
const docs = await loader.load();
```

## Text Splitters

Text splitters break documents into manageable chunks while preserving context.

```javascript
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 50,
});

const splitDocs = await textSplitter.splitDocuments(docs);
```

## Embeddings

Convert text into vector representations for semantic search.

```javascript
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';

const embeddings = new OpenAIEmbeddings();
```

## Vector Stores

Store and retrieve vector embeddings efficiently.

```javascript
import { FaissStore } from 'langchain/vectorstores/faiss';

const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);
const retriever = vectorStore.asRetriever();
```

## Retrieval Chain

Combine retrieval with LLM responses.

```javascript
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { RetrievalQAChain } from 'langchain/chains';

const model = new ChatOpenAI({ modelName: 'gpt-4', temperature: 0 });
const chain = RetrievalQAChain.fromLLM(model, retriever);

const response = await chain.call({
  query: 'What information do you need?'
});
```

## LangChain Expression Language (LCEL)

Modern approach using composable runnables:

```javascript
import { RunnableSequence, RunnableParallel } from 'langchain/schema/runnable';

const retrieverChain = RunnableSequence.from([
  RunnableParallel({
    context: retriever,
    question: input => input.question
  }),
  prompt,
  model,
  output => output.text
]);
```

## Best Practices

1. Use appropriate chunk sizes for your use case
2. Consider overlap between chunks for context preservation
3. Choose the right vector store for your scale
4. Set appropriate similarity thresholds
5. Use temperature=0 for consistent retrievals

## Tips for Implementation

- Start with small document sets for testing
- Monitor token usage and embedding costs
- Consider caching for frequently accessed documents
- Use filtering when possible to improve relevance

# Chains in LangChain

## Overview
Chains are end-to-end wrappers around multiple components executed in a defined order. They enable:
- Breaking complex tasks into smaller steps
- Combining different models and utilities
- Adding state and memory between calls
- Simplifying debugging

## Basic Chain Types

### LLM Chain
Simple chain passing prompts to a language model:

```javascript
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { ChatPromptTemplate } from 'langchain/prompts';
import { StringOutputParser } from 'langchain/schema/output_parser';

const model = new ChatOpenAI({ modelName: 'gpt-4' });
const prompt = ChatPromptTemplate.fromMessages([
  ['system', 'You are PaRappa the Rapper. You spit hot fire flows like lava.'],
  ['human', 'Write a comedic, parody rap about the following topic: {topic}']
]);

const chain = prompt.pipe(model).pipe(new StringOutputParser());

// For streaming output
const stream = await chain.stream({
  topic: 'artificial intelligence'
});
```

### Router Chain
Route inputs to different chains based on conditions:

```javascript
import { RunnableBranch } from 'langchain/schema/runnable';

const promptBranch = new RunnableBranch()
  .addBranch((x) => x.includes('history'), historyPrompt)
  .addBranch((x) => x.includes('science'), sciencePrompt)
  .setDefaultBranch(generalPrompt);

const chain = classifier
  .pipe(promptBranch)
  .pipe(model)
  .pipe(new StringOutputParser());
```

### Sequential Chain
Chain multiple operations where output of one feeds into another:

```javascript
const sequentialChain = promptTemplate1
  .pipe(model)
  .pipe(new StringOutputParser())
  .pipe(promptTemplate2)
  .pipe(model)
  .pipe(new StringOutputParser());

const response = await sequentialChain.invoke({
  topic: "initial input"
});
```

### Transform Chain
Modify inputs between chain steps:

```javascript
const transformChain = (text) => text.substring(0, 1000);

const chain = transformChain
  .pipe(prompt)
  .pipe(model)
  .pipe(new StringOutputParser());
```

## Best Practices for Chains

1. Use LCEL (LangChain Expression Language) for modern chain construction
2. Implement proper error handling between chain steps
3. Consider using streaming for better user experience
4. Break complex chains into smaller, manageable pieces
5. Add appropriate logging and monitoring

## Tips for Chain Implementation

- Start with simple chains and gradually add complexity
- Test each chain component independently
- Use streaming for long-running chains
- Consider implementing retry logic for failed steps
- Monitor token usage across chain steps

# Agents in LangChain

## Overview
Agents are systems that use language models to interact with tools. Unlike chains which have hard-coded sequences, agents:
- Use LLMs to choose action sequences dynamically
- Interact with tools and environments
- Make decisions based on observations
- Maintain context through memory

## Core Components

### Tools and Toolkits
Tools are individual components for specific tasks, while toolkits are collections of related tools.

```javascript
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { DuckDuckGoSearchTool } from 'langchain/tools/ddg';
import { WikipediaQueryRun } from 'langchain/tools/wikipedia';
import { Calculator } from 'langchain/tools/calculator';

const model = new ChatOpenAI({ temperature: 0 });
const tools = [
  new DuckDuckGoSearchTool(),
  new WikipediaQueryRun(),
  new Calculator(),
];
```

### Basic Agent Setup (Legacy v0.1)

```javascript
import { initializeAgentExecutorWithOptions } from 'langchain/agents';

const executor = await initializeAgentExecutorWithOptions(
  tools,
  model,
  {
    agentType: 'zero-shot-react-description',
  }
);

const result = await executor.run(
  'Who is the current chief AI scientist at Meta? What's the temperature there?'
);
```

### Modern Agent Setup (v0.2+)

```javascript
import { 
  createOpenAIFunctionsAgent,
  AgentExecutor 
} from 'langchain/agents';

const agent = await createOpenAIFunctionsAgent({
  llm: model,
  tools,
  prompt: agentPrompt
});

const agentExecutor = new AgentExecutor({
  agent,
  tools
});
```

### Agents with Memory

```javascript
import { BufferMemory } from 'langchain/memory';
import { MessagesPlaceholder } from 'langchain/prompts';

const memory = new BufferMemory({
  returnMessages: true,
  memoryKey: 'chat_history'
});

const agentWithMemory = AgentExecutor.fromAgentAndTools({
  agent,
  tools,
  memory
});

// First query establishes context
await agentWithMemory.invoke({
  input: "What's the weather in Winnipeg?"
});

// Follow-up maintains context
await agentWithMemory.invoke({
  input: "What should I wear then?"
});
```

## Best Practices for Agents

1. Set temperature=0 for structured tasks
2. Choose appropriate tools for the task
3. Implement proper error handling
4. Use memory for contextual conversations
5. Consider rate limits of external tools

## Tips for Agent Implementation

- Start with simple tools and gradually add complexity
- Test agent behavior with various inputs
- Monitor tool usage and API calls
- Consider implementing timeouts
- Use streaming for better user experience

# Memory in LangChain

## Overview
Memory in LangChain persists state between chain or agent calls, enabling context retention across interactions. Like Dory from "Finding Nemo", LLMs naturally have short-term memory - the Memory module helps solve this limitation.

Key functions:
- Reads and writes conversation history
- Supplements user input with context
- Stores current interactions for future use

## Memory Types

### Conversation Buffer Memory
Simple memory that stores complete chat history.

```javascript
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { ConversationBufferMemory } from 'langchain/memory';
import { MessagesPlaceholder } from 'langchain/prompts';

const memory = new ConversationBufferMemory({
  returnMessages: true,
  memoryKey: "history"
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful AI assistant"],
  new MessagesPlaceholder("history"),
  ["human", "{input}"]
]);

const chain = prompt
  .pipe(new ChatOpenAI())
  .bind({ history: memory });

// Use the chain
const response1 = await chain.invoke({ 
  input: "Hi, I'm John"
});
const response2 = await chain.invoke({ 
  input: "What's my name?"
}); // Will remember "John"
```

### Conversation Buffer Window Memory
Maintains a sliding window of most recent interactions.

```javascript
import { ConversationBufferWindowMemory } from 'langchain/memory';

const windowMemory = new ConversationBufferWindowMemory({
  k: 3, // Keep last 3 interactions
  returnMessages: true
});

const chain = new ConversationChain({
  llm: model,
  memory: windowMemory
});
```

### Conversation Summary Memory
Condenses conversation history into summaries.

```javascript
import { ConversationSummaryMemory } from 'langchain/memory';

const summaryMemory = new ConversationSummaryMemory({
  llm: new ChatOpenAI({ temperature: 0 }),
  returnMessages: true
});

const chain = new ConversationChain({
  llm: model,
  memory: summaryMemory
});
```

## Memory Usage Patterns

### With LCEL (Modern Approach)
```javascript
const chain = RunnableSequence.from([
  {
    context: memory.loadMemoryVariables.bind(memory),
    input: input => input.question
  },
  prompt,
  model,
  new StringOutputParser()
]);
```

### With Agents
```javascript
const agent = AgentExecutor.fromAgentAndTools({
  agent,
  tools,
  memory: new BufferMemory({
    memoryKey: "chat_history"
  })
});
```

## Best Practices

1. Choose appropriate memory type for your use case:
   - Buffer Memory: Short conversations
   - Window Memory: Long conversations with recent context
   - Summary Memory: Extended conversations needing condensed context

2. Consider token limits:
   - Monitor context window size
   - Use window or summary memory for long conversations
   - Clear memory when appropriate

3. Memory Management:
   - Initialize memory at start of conversation
   - Save important context explicitly
   - Clear memory when conversation ends

## Tips for Implementation

- Test memory retention with follow-up questions
- Monitor token usage with different memory types
- Consider persistence for long-running applications
- Use memory selectively - not all chains need it
- Implement memory clearing mechanisms

# Prompt Engineering Basics

## Overview
Modern language models act as a Swiss Army knife for various language tasks through the power of prompting. Success with LLMs largely depends on effective prompt engineering and in-context learning.

## Core Concepts

### In-Context Learning
- Method to guide LLM behavior without model weight updates
- Effectiveness varies across different models
- Requires experimentation and iteration

### Elements of an Effective Prompt

1. **Instruction**
   - Clear directive for the model's task
   - Example: "Classify the following text as positive or negative"

2. **Context**
   - Background information to refine responses
   - Can be manual or from external sources
   ```javascript
   const context = `
   Company background:
   Founded in 2020
   Industry: Tech
   Revenue: $10M
   `;
   ```

3. **Input Data**
   - Specific question or topic
   - Example: "What are our company's strengths?"

4. **Output Indicator**
   - Desired response format
   ```javascript
   const prompt = `
   ${context}
   Question: ${input}
   Please provide your answer in bullet points, focusing on key metrics.
   `;
   ```

## Best Practices

1. **Prompt Structure**
   - Place instructions towards the end
   - Keep context concise and relevant
   - Be specific about desired output format

2. **Testing and Iteration**
   - Try different prompt variations
   - Test across different inputs
   - Monitor and adjust based on results

3. **Common Patterns**
   ```javascript
   // Basic prompt template
   const basicPrompt = `
   Context: ${context}
   Input: ${userInput}
   Instructions: ${task}
   Output Format: ${format}
   `;

   // Role-based prompt
   const rolePrompt = `
   You are an expert ${role}.
   Given this context: ${context}
   ${task}
   `;
   ```

## Tips for Implementation

- Start simple, then add complexity
- Test prompts with edge cases
- Consider token limits
- Balance between specificity and flexibility
- Document successful prompt patterns

## Example Implementations

```javascript
// Basic prompt implementation
const getResponse = async (task, context, input) => {
  const prompt = `
  Context: ${context}
  Input: ${input}
  Task: ${task}
  Provide your response in a clear, structured format.
  `;
  
  return await model.invoke(prompt);
};

// Role-based prompt with specific output format
const getStructuredResponse = async (role, task, format) => {
  const prompt = `
  As an expert ${role},
  ${task}
  
  Provide your response in the following format:
  ${format}
  `;
  
  return await model.invoke(prompt);
};
```

# Prompting Principles and Tactics

## Core Principles

### 1. Clarity
- Avoid ambiguity
- Use delimiters to separate instructions from content
- Consider model-specific tokens

```javascript
// Example with delimiters and instruction tokens
const prompt = `[INST]
Summarize the following text:
"""
${text}
"""
Keep the summary concise and focused on key points.
[/INST]`;
```

### 2. Structured Output
Request specific formats for easier parsing:

```javascript
const structuredPrompt = `
Create a movie entry with the following JSON structure:
{
  "id": number,
  "title": string,
  "director": string,
  "genre": string
}

Ensure the output is valid JSON.`;

// Example with conditional structure
const conditionalPrompt = `
Analyze the text between <text></text> tags.
If it contains instructions:
- Title: [procedure name]
- Steps: [numbered list]
If not:
- Response: "No procedural guide found"

<text>${input}</text>`;
```

### 3. Complex Query Handling

1. **Step-by-Step Reasoning**
```javascript
const complexPrompt = `
Solve this problem step by step:
1. First, work out the solution yourself
2. Then, compare with the given solution
3. Finally, determine correctness

Problem: ${problem}
Student Solution: ${solution}

Format your response as:
- Your Solution: [steps]
- Comparison: [analysis]
- Verdict: [correct/incorrect]`;
```

2. **Thought Process Guidance**
```javascript
const reasoningPrompt = `
Before providing the final answer:
1. List key information
2. Show your reasoning
3. Draw conclusions

Question: ${question}

Structure your response:
- Given Information:
- Analysis:
- Conclusion:`;
```

### 4. Context Balance
- Prioritize essential information
- Avoid context overload
- Let model request additional context if needed

```javascript
const balancedPrompt = `
Primary Context: ${mainContext}
Task: ${specificTask}

If you need additional context, please request it.
Otherwise, proceed with the task using available information.`;
```

## Implementation Tactics

1. **Model-Specific Considerations**
```javascript
// For smaller models (e.g., Mistral-7B)
const mistralPrompt = `[INST]${instruction}[/INST]`;

// For OpenAI models
const openAIPrompt = `${instruction}`;
```

2. **Verification Patterns**
```javascript
const verificationPrompt = `
Verify the following criteria:
1. ${condition1}
2. ${condition2}

For each criterion:
- State if it's met
- Provide reasoning
- Suggest improvements if needed`;
```

## Best Practices Summary

1. **Clarity**
   - Use clear delimiters
   - Provide explicit instructions
   - Separate context from commands

2. **Structure**
   - Request specific output formats
   - Use conditional responses
   - Define clear response templates

3. **Complex Tasks**
   - Break down into steps
   - Request explicit reasoning
   - Allow for iterative solutions

4. **Context Management**
   - Provide essential context first
   - Allow for follow-up questions
   - Balance detail with clarity

# Prompt Templates

## Overview
Prompt templates are tools for producing prompts consistently and repeatably. They combine:
- Instructions
- External context
- User input/query
- Output indicators

## Basic Implementation

### Single Input Variable Templates
```javascript
import { PromptTemplate } from 'langchain/prompts';

// Method 1: Using constructor
const template1 = new PromptTemplate({
  template: "Answer the following question: {query}",
  inputVariables: ["query"]
});

// Method 2: Recommended - Using from_template
const template2 = PromptTemplate.fromTemplate(
  "Answer the following question: {query}"
);

// Usage with LangChain Expression Language (LCEL)
const chain = template2
  .pipe(new ChatOpenAI())
  .pipe(new StringOutputParser());

const response = await chain.invoke({
  query: "Why is a SoftMax function used in NNs?"
});
```

### Reusable Advice Function Example
```javascript
const getAdvice = async (topic) => {
  const model = new ChatOpenAI({ 
    temperature: 0.7 
  });
  
  const template = PromptTemplate.fromTemplate(
    "Give me advice about: {topic}. Be concise and practical."
  );
  
  const chain = template
    .pipe(model)
    .pipe(new StringOutputParser());
    
  return await chain.stream({
    topic: topic
  });
};

// Usage
await getAdvice("balancing priorities");
```

## Best Practices

1. **Template Creation**
   - Prefer `fromTemplate()` over constructor
   - Keep templates reusable and modular
   - Document input variables

2. **Variable Management**
   - Use descriptive variable names
   - Consider default values when appropriate
   - Validate required inputs

3. **Integration**
   - Use with LCEL for modern pipelines
   - Consider streaming for better UX
   - Chain with appropriate output parsers

## Common Patterns

```javascript
// Generic QA Template
const qaTemplate = PromptTemplate.fromTemplate(`
Context: {context}
Question: {question}
Provide a clear and concise answer based on the context.
`);

// Role-based Template
const expertTemplate = PromptTemplate.fromTemplate(`
You are an expert in {field}.
Question: {query}
Provide your expert analysis and explanation.
`);

// Multiple Input Template
const analysisTemplate = PromptTemplate.fromTemplate(`
Compare {item1} with {item2} considering:
1. Similarities
2. Differences
3. Use cases

Format as bullet points.
`);
```

## Tips for Template Design

- Keep templates focused and single-purpose
- Include clear formatting instructions
- Consider model context limits
- Test templates with edge cases
- Use appropriate delimiters for variables

# Multi-Input Prompt Templates

## Basic Implementation

### Legacy Approach (Constructor Method)
```javascript
import { PromptTemplate } from 'langchain/prompts';
import { ChatOpenAI } from 'langchain/chat_models/openai';

const createMoviePrompt = () => {
  return new PromptTemplate({
    template: "Create a synopsis and genre for a fictional movie titled '{movie}' starring {actor}.",
    inputVariables: ["movie", "actor"]
  });
};

const formattedPrompt = await prompt.format({
  movie: "Jatt da Pajama Uuchaa Ho Gayaa",
  actor: "AP Dhillon"
});
```

### Modern Approach (LCEL)
```javascript
// Recommended method using fromTemplate
const promptTemplate = PromptTemplate.fromTemplate(
  "Create a synopsis and genre for a fictional movie titled '{movie}' starring {actor}."
);

const model = new ChatOpenAI({ temperature: 0.7 });

const chain = promptTemplate
  .pipe(model)
  .pipe(new StringOutputParser());

// Non-streaming usage
const response = await chain.invoke({
  movie: "Amritsar: 1984",
  actor: "Gurdaas Mann"
});

// Streaming usage
for await (const chunk of chain.stream({
  movie: "Amritsar: 1984",
  actor: "Gurdaas Mann"
})) {
  console.log(chunk);
}
```

## Complex Template Examples

```javascript
// Movie Analysis Template
const movieAnalysisTemplate = PromptTemplate.fromTemplate(`
Movie: {movie}
Actor: {actor}
Director: {director}

Please provide:
1. Plot synopsis
2. Genre analysis
3. Target audience
4. Box office potential
`);

// Comparison Template
const comparisonTemplate = PromptTemplate.fromTemplate(`
Compare these two movies:

Movie 1: {movie1}
Starring: {actor1}
Genre: {genre1}

Movie 2: {movie2}
Starring: {actor2}
Genre: {genre2}

Analyze:
1. Thematic similarities
2. Target audience overlap
3. Market positioning
`);
```

## Best Practices for Multi-Input Templates

1. **Variable Organization**
   - Group related variables together
   - Use descriptive naming conventions
   - Consider optional variables

2. **Input Validation**
   - Validate all required inputs
   - Provide default values when appropriate
   - Handle missing inputs gracefully

3. **Performance Optimization**
   - Use streaming for large outputs
   - Consider batching related requests
   - Cache frequently used templates

## Implementation Tips

- Keep template structure consistent
- Document input requirements clearly
- Test with various input combinations
- Consider error handling for missing inputs
- Use type checking when possible

# Chat Prompt Templates

## Overview
Chat prompt templates work with a list of messages, where each message has:
- Content
- Role (system, human, or AI assistant)
- Optional input variables

## Basic Implementation

### Using Message Tuples
```javascript
import { ChatPromptTemplate } from 'langchain/prompts';
import { ChatOpenAI } from 'langchain/chat_models/openai';

const chatPrompt = ChatPromptTemplate.fromMessages([
  ['system', 'You are a helpful yet quirky AI bot. Your name is {name}.'],
  ['human', '{user_input}'],
  ['ai', 'I will help you with that!']
]);

const model = new ChatOpenAI({ temperature: 0.8 });

const chain = chatPrompt
  .pipe(model)
  .pipe(new StringOutputParser());

// Usage
await chain.invoke({
  name: "Robo Talker",
  user_input: "talk robo to me"
});
```

### Using Message Templates
```javascript
import { 
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  AIMessagePromptTemplate
} from 'langchain/prompts';

const systemTemplate = SystemMessagePromptTemplate.fromTemplate(
  "You are a {personality} AI assistant. Be {style} in your responses."
);

const humanTemplate = HumanMessagePromptTemplate.fromTemplate(
  "{text}"
);

const chatPrompt = ChatPromptTemplate.fromMessages([
  systemTemplate,
  humanTemplate
]);

// Using the template
const chain = chatPrompt
  .pipe(new ChatOpenAI())
  .pipe(new StringOutputParser());

await chain.stream({
  personality: "sarcastic",
  style: "super freaking rude",
  text: "Tell me about your day"
});
```

## Best Practices

1. **Message Structure**
   - Keep system messages clear and consistent
   - Use appropriate roles for each message
   - Consider message order impact

2. **Template Organization**
   - Separate system and user templates
   - Maintain consistent personality
   - Use descriptive variable names

3. **Role Management**
   - Use system messages for core instructions
   - Use human messages for user inputs
   - Use AI messages for example responses

## Implementation Tips

- Test different temperature settings
- Consider streaming for interactive chats
- Document role expectations
- Handle multi-turn conversations
- Include error handling for missing variables

# Prompt Serialization

## Overview
Prompt serialization in LangChain enables efficient storage and sharing of prompts using human-readable formats (JSON/YAML). Key features:
- Human-readable storage formats
- Flexible storage options (single/multiple files)
- Unified loading interface
- Version control friendly

## Basic Implementation

### Saving Prompts
```javascript
import { PromptTemplate } from 'langchain/prompts';

const prompt = new PromptTemplate({
  template: "Tell me a {adjective} story about {noun}",
  inputVariables: ["adjective", "noun"]
});

// Save as JSON
const serialized = prompt.toJSON();
console.log(JSON.stringify(serialized, null, 2));

// Save to file
import { writeFileSync } from 'fs';
writeFileSync('prompt.json', JSON.stringify(serialized, null, 2));
```

### Loading Prompts
```javascript
import { PromptTemplate } from 'langchain/prompts';
import { loadPrompt } from 'langchain/prompts/load';

// Load from file
const loadedPrompt = await loadPrompt('prompt.json');

// Load from Hub
const hubPrompt = await loadPrompt(
  'lc://prompts/conversation/summarize/prompt.json'
);
```

## Best Practices

1. **File Organization**
   - Use consistent naming conventions
   - Group related prompts together
   - Consider versioning strategy

2. **Format Selection**
   - JSON for programmatic access
   - YAML for human readability
   - Structure for reusability

3. **Version Control**
   - Track prompt changes
   - Document prompt versions
   - Consider using LangChain Hub

## Implementation Tips

- Validate prompts after loading
- Consider environment-specific variations
- Document prompt dependencies
- Use descriptive filenames
- Include metadata in serialized prompts

# Zero-Shot Prompting

## Overview
Zero-shot prompting allows language models to perform tasks without specific examples or training, leveraging their pre-training on vast datasets.

## Basic Implementation

```javascript
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PromptTemplate } from 'langchain/prompts';
import { StringOutputParser } from 'langchain/schema/output_parser';

class ZeroShotChain {
  constructor(model) {
    this.model = model;
  }

  async classify(template, input) {
    const prompt = PromptTemplate.fromTemplate(template);
    const chain = prompt
      .pipe(this.model)
      .pipe(new StringOutputParser());
    
    return await chain.invoke({ input });
  }
}

// Usage Examples
const model = new ChatOpenAI({ temperature: 0 });
const zeroShot = new ZeroShotChain(model);

// Sarcasm Detection
const sarcasmResult = await zeroShot.classify(
  "Classify if the text in [brackets] is sarcastic or not sarcastic: [{input}]",
  "Oh great, another meeting that could have been an email!"
);

// Named Entity Recognition
const nerResult = await zeroShot.classify(
  "Identify and categorize names, locations, and organizations in this text: [{input}]",
  "Barack Obama was born in Honolulu and later became CEO of the United States."
);
```

## Common Use Cases

1. **Text Classification**
   ```javascript
   const template = `
   Classify the following text as {categories}:
   Text: {input}
   Classification:`;
   ```

2. **Entity Recognition**
   ```javascript
   const template = `
   Extract and categorize these entity types from the text:
   - Names
   - Locations
   - Organizations
   
   Text: {input}
   
   Entities:`;
   ```

3. **Content Generation**
   ```javascript
   const template = `
   Write a {style} {content_type} using this information:
   {input}
   
   Output:`;
   ```

## Limitations and Best Practices

1. **When to Use**
   - Simple classification tasks
   - Basic entity extraction
   - General knowledge queries
   - Standard text transformations

2. **When to Avoid**
   - Complex, nuanced tasks
   - Domain-specific jargon
   - Tasks requiring precise control
   - Large text processing

3. **Optimization Tips**
   - Keep instructions clear and concise
   - Use specific output formats
   - Consider task complexity
   - Have fallback strategies

## Implementation Tips

- Start with simple tasks
- Validate outputs thoroughly
- Use structured output formats
- Monitor performance across different inputs
- Consider few-shot alternatives for complex tasks

# Custom Prompt Templates

## Overview
Custom prompt templates provide dynamic prompt generation capabilities beyond simple string substitutions. They're useful when you need:
- Unique instructions or context
- Dynamic task-specific information
- Full control over prompt structure

## Basic Implementation

### Function Explainer Template
```javascript
import { StringPromptTemplate } from 'langchain/prompts';
import { BasePromptTemplate } from 'langchain/prompts/base';

class FunctionExplainerPromptTemplate extends StringPromptTemplate {
  constructor() {
    super({
      template: `Given this function name and source code, explain what it does:
Function Name: {function_name}
Source Code:
{source_code}
      
Explanation:`,
      inputVariables: ["function_name"]
    });
  }

  async format(values) {
    const { function_name } = values;
    const source_code = this.getSourceCode(function_name);
    return super.format({ 
      function_name, 
      source_code 
    });
  }

  private getSourceCode(fn) {
    return fn.toString();
  }

  _getPromptType() {
    return "function-explainer";
  }
}

// Usage with LCEL
const fnExplainer = new FunctionExplainerPromptTemplate();
const chain = fnExplainer
  .pipe(new ChatOpenAI())
  .pipe(new StringOutputParser());

await chain.stream({
  function_name: myFunction
});
```

### Algorithm Optimizer Template
```javascript
class AlgorithmOptimizerPromptTemplate extends StringPromptTemplate {
  constructor() {
    super({
      template: `Analyze and optimize this algorithm:
{algorithm_code}

Provide:
1. Time complexity analysis
2. Space complexity analysis
3. Optimization suggestions`,
      inputVariables: ["algorithm_code"]
    });
  }

  async format(values) {
    const { algorithm_code } = values;
    return super.format({ 
      algorithm_code: algorithm_code.toString() 
    });
  }

  _getPromptType() {
    return "algorithm-optimizer";
  }
}
```

## Best Practices

1. **Template Design**
   - Clearly define input variables
   - Implement robust format methods
   - Include validation logic
   - Consider error handling

2. **Code Organization**
   - Extend appropriate base classes
   - Keep methods focused
   - Document template behavior
   - Include type definitions

3. **Integration**
   - Test with various inputs
   - Consider edge cases
   - Optimize for reusability
   - Use with LCEL when possible

## Implementation Tips

- Start with base template classes
- Include proper validation
- Document expected inputs
- Consider performance implications
- Test with various input types

# Prompt Pipelining

## Overview
Prompt pipelining enables modular and efficient prompt design by:
- Reusing prompt components
- Splitting long prompts into logical chunks
- Dynamically assembling prompts
- Mixing static text with templates
- Building composable chat prompts

## String Prompt Pipelining

```javascript
import { PromptTemplate } from 'langchain/prompts';

// Basic Pipeline
const baseTemplate = PromptTemplate.fromTemplate(
  "I'm heading to {destination}, recommend a great {activity}."
);
const additionalTemplate = PromptTemplate.fromTemplate(
  "Also any local delicacies I should try?"
);

const pipeline = baseTemplate + additionalTemplate;

// Adding More Components
const greetingTemplate = PromptTemplate.fromTemplate(
  "How do I greet the locals in a jolly, informal manner?"
);
const fullPrompt = pipeline + greetingTemplate;

// Usage with Chain
const chain = fullPrompt
  .pipe(new ChatOpenAI())
  .pipe(new StringOutputParser());

await chain.invoke({
  destination: "Tokyo",
  activity: "evening activity"
});
```

### Travel Chatbot Example
```javascript
class TravelChatbot {
  private prompt: PromptTemplate;

  constructor(basePrompt: PromptTemplate) {
    this.prompt = basePrompt;
  }

  addQuestion(template: PromptTemplate) {
    this.prompt = this.prompt + template;
    return this;
  }

  async getResponse(inputs: Record<string, string>) {
    const chain = this.prompt
      .pipe(new ChatOpenAI())
      .pipe(new StringOutputParser());
    
    return await chain.invoke(inputs);
  }
}

// Usage
const bot = new TravelChatbot(baseTemplate)
  .addQuestion(additionalTemplate)
  .addQuestion(greetingTemplate);

await bot.getResponse({
  destination: "Paris",
  activity: "morning walk"
});
```

## Key Differences from Multi-Input Prompts

1. **Composition vs Input Handling**
   - Pipelining: Composes multiple prompts sequentially
   - Multi-input: Handles multiple inputs in one template

2. **Component Management**
   - Pipelining: Independent formatting of components
   - Multi-input: Single template formatting

3. **Reusability**
   - Pipelining: Reuse components across prompts
   - Multi-input: Focus on input handling

## Best Practices

1. **Design Principles**
   - Keep components modular
   - Use clear component boundaries
   - Consider component dependencies

2. **Performance**
   - Cache reusable components
   - Monitor pipeline complexity
   - Consider prompt length limits

3. **Maintenance**
   - Document component purposes
   - Version control components
   - Test pipeline combinations

## Implementation Tips

- Start with base templates
- Build incrementally
- Test component combinations
- Consider error handling
- Monitor token usage

# Chat Prompt Pipelining

## Overview
Chat prompt pipelining enables creation of chat prompts using reusable message components where:
- Each pipeline addition becomes a new message
- Static messages integrate with templated messages
- Components are easily reusable
- Prompts construct dynamically

## Basic Implementation

```javascript
import { 
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  AIMessagePromptTemplate
} from 'langchain/prompts';

// System Message Setup
const systemMessage = SystemMessagePromptTemplate.fromTemplate(
  "Welcome to the East End Cockney Chat. {style}"
);

// Building the Pipeline
const chatPrompt = ChatPromptTemplate.fromMessages([
  systemMessage,
  HumanMessagePromptTemplate.fromTemplate("Oi mate! {query}"),
  AIMessagePromptTemplate.fromTemplate("Cor blimey! {response}"),
  HumanMessagePromptTemplate.fromTemplate("{input}")
]);

// Usage with Chain
const chain = chatPrompt
  .pipe(new ChatOpenAI())
  .pipe(new StringOutputParser());

const response = await chain.invoke({
  style: "Keep it proper Cockney",
  query: "What's the weather like?",
  response: "Bit cloudy innit?",
  input: "No, what about it?"
});
```

## Message Component Types

```javascript
// System Instructions
const systemInstructions = SystemMessagePromptTemplate.fromTemplate(
  "You are a {personality} AI assistant. {guidelines}"
);

// Human Messages
const userQuery = HumanMessagePromptTemplate.fromTemplate(
  "{question}"
);

// AI Responses
const aiResponse = AIMessagePromptTemplate.fromTemplate(
  "{response_format}: {content}"
);

// Combining Components
const fullPrompt = ChatPromptTemplate.fromMessages([
  systemInstructions,
  userQuery,
  aiResponse,
  userQuery
]);
```

## Best Practices

1. **Message Organization**
   - Keep system messages first
   - Group related messages
   - Maintain conversation flow

2. **Template Design**
   - Use clear variable names
   - Consider message order
   - Keep templates focused

3. **Pipeline Management**
   - Reuse common components
   - Version message templates
   - Test conversation flows

## Implementation Tips

- Start with system context
- Build conversation flow logically
- Test with various inputs
- Consider message dependencies
- Monitor prompt length

# Prompt Composition

## Overview
Prompt composition enables reuse of prompt parts through pipeline prompts, consisting of:
- Final prompt template
- List of pipeline prompts (name-template pairs)
- Variable passing between templates

## Basic Implementation

```javascript
import { 
  PipelinePromptTemplate,
  PromptTemplate 
} from 'langchain/prompts';

// Final Prompt Template
const fullPrompt = PromptTemplate.fromTemplate(`
{introduction}
{example}
{start}
`);

// Component Templates
const introTemplate = PromptTemplate.fromTemplate(
  "Let's have a conversation with {person}"
);

const exampleTemplate = PromptTemplate.fromTemplate(`
Q: {example_q}
A: {example_a}
`);

const startTemplate = PromptTemplate.fromTemplate(
  "Now answer this: {input}"
);

// Composing Pipeline
const pipelinePrompt = new PipelinePromptTemplate({
  finalPrompt: fullPrompt,
  pipelinePrompts: [
    ["introduction", introTemplate],
    ["example", exampleTemplate],
    ["start", startTemplate]
  ]
});

// Usage
const chain = pipelinePrompt
  .pipe(new ChatOpenAI())
  .pipe(new StringOutputParser());

const response = await chain.invoke({
  person: "Elon Musk",
  example_q: "What's your favorite social media?",
  example_a: "Twitter, of course!",
  input: "Why that platform?"
});
```

## Cooking Chatbot Example

```javascript
class CookingChatbot {
  private prompt: PipelinePromptTemplate;

  constructor() {
    const finalTemplate = PromptTemplate.fromTemplate(`
      {intro}
      {style}
      {question}
    `);

    const introTemplate = PromptTemplate.fromTemplate(
      "Welcome to Chef's Kitchen! {chef_name} will guide you."
    );

    const styleTemplate = PromptTemplate.fromTemplate(
      "Cooking style: {cuisine_type}"
    );

    const questionTemplate = PromptTemplate.fromTemplate(
      "Q: {query}"
    );

    this.prompt = new PipelinePromptTemplate({
      finalPrompt: finalTemplate,
      pipelinePrompts: [
        ["intro", introTemplate],
        ["style", styleTemplate],
        ["question", questionTemplate]
      ]
    });
  }

  async getResponse(inputs: Record<string, string>) {
    const chain = this.prompt
      .pipe(new ChatOpenAI())
      .pipe(new StringOutputParser());
    
    return await chain.invoke(inputs);
  }
}
```

## Best Practices

1. **Template Organization**
   - Separate reusable components
   - Name templates descriptively
   - Consider component dependencies

2. **Variable Management**
   - Track variable flow between templates
   - Use clear naming conventions
   - Document variable requirements

3. **Performance Considerations**
   - Note streaming limitations
   - Consider template complexity
   - Cache repeated components

## Implementation Tips

- Plan component hierarchy
- Document variable flow
- Consider template reusability
- Test component combinations
- Handle composition errors

# Few-Shot Prompt Templates

## Overview
Few-shot learning is a form of in-context learning where you provide a few examples in the prompt to prime the model for your task. LangChain's `FewShotPromptTemplate` lets you supply sample input/output pairs to guide the model's behavior.

## Basic Implementation

```javascript
import { FewShotPromptTemplate, PromptTemplate } from 'langchain/prompts';

// Example prompt for each sample
const examplePrompt = PromptTemplate.fromTemplate(
  "Question: {question}\nAnswer: {answer}\n"
);

// List of examples
const examples = [
  { question: "What is the capital of France?", answer: "Paris" },
  { question: "Who wrote Hamlet?", answer: "William Shakespeare" },
  { question: "What is the boiling point of water?", answer: "100°C" }
];

// Few-shot prompt template
const fewShotPrompt = new FewShotPromptTemplate({
  examples,
  examplePrompt,
  prefix: "Answer the following questions:",
  suffix: "Question: {input}\nAnswer:",
  inputVariables: ["input"]
});

// Usage with chain
const chain = fewShotPrompt
  .pipe(new ChatOpenAI({ temperature: 0.1 }))
  .pipe(new StringOutputParser());

const response = await chain.invoke({
  input: "What is the largest planet in our solar system?"
});
```

## Dynamic Few-Shot Example Selection

```javascript
// Function to generate prefix and select random examples from a dataset
function getPrefixAndExamples(dataset, intentKey = "intent", textKey = "text") {
  const intents = [...new Set(dataset.map(item => item[intentKey]))];
  const prefix = `Each input is associated with a user intent. Choose from the following intents: ${intents.join(", ")}.\n`;
  // Select one random example per intent
  const examples = intents.map(intent => {
    const filtered = dataset.filter(item => item[intentKey] === intent);
    const example = filtered[Math.floor(Math.random() * filtered.length)];
    return { text: example[textKey], intent: example[intentKey] };
  });
  return { prefix, examples };
}

// Example usage
const dataset = [
  { text: "Save Free Smoke by AP Dhillon to my songs.", intent: "add_to_playlist" },
  { text: "Play the next song.", intent: "play_next" },
  { text: "Pause the music.", intent: "pause" }
];

const { prefix, examples } = getPrefixAndExamples(dataset);

const examplePrompt = PromptTemplate.fromTemplate(
  "Text: {text}\nIntent: {intent}\n"
);

const fewShotIntentPrompt = new FewShotPromptTemplate({
  examples,
  examplePrompt,
  prefix,
  suffix: "Text: {input}\nIntent:",
  inputVariables: ["input"]
});

const chain = fewShotIntentPrompt
  .pipe(new ChatOpenAI({ temperature: 0.1 }))
  .pipe(new StringOutputParser());

const response = await chain.invoke({
  input: "Save Free Smoke by AP Dhillon to my songs."
});
```

## Best Practices

1. **Example Selection**
   - Use relevant, diverse examples
   - Match style and format to desired output
   - Limit number of examples for context window

2. **Template Design**
   - Keep example prompts concise
   - Use clear input/output indicators
   - Maintain consistent formatting

3. **Dynamic Example Generation**
   - Select examples programmatically for large datasets
   - Randomize or stratify examples for coverage

## Implementation Tips

- Use FewShotPromptTemplate for tasks needing context or style priming
- Test with different numbers and types of examples
- Monitor token usage with large example sets
- Engineer prompts for specific tasks and domains

# Few-Shot Chat Prompt Templates

## Overview
Few-shot chat examples assist in steering chat models by presenting sample dialogues that exhibit the desired patterns, flow, and tone of conversation.

The primary distinctions from standard few-shot examples are:
- They are structured as dialogues.
- They simulate a conversation between two participants.
- They emphasize the flow and tone of the conversation.

## Basic Implementation

```javascript
import { 
  ChatPromptTemplate,
  FewShotChatMessagePromptTemplate,
  SystemMessagePromptTemplate
} from 'langchain/prompts';

// Sample dialogues
const examples = [
  {
    input: "What transpired during the Enchantment Under the Sea dance?",
    response: "That's the event where Marty's parents, George and Lorraine, shared their first kiss to the tune of 'Earth Angel'. This moment was vital for Marty's existence!"
  },
  {
    input: "What food do you enjoy the most?",
    response: "🎵 The power of love is a mysterious force, It can make one man cry, and another man sing 🎵"
  }
];

// Construct system message
const systemMessage = SystemMessagePromptTemplate.fromTemplate(
  "You're a bot that knows everything about Back to the Future. Only respond to inquiries related to Back to the Future. For questions off-topic, quote lyrics from the soundtrack."
);

// Develop few-shot template
const fewShotTemplate = FewShotChatMessagePromptTemplate.fromExamples(
  examples,
  {
    humanMessageTemplate: "{input}",
    aiMessageTemplate: "{response}"
  }
);

// Merge into the final prompt
const chatPrompt = ChatPromptTemplate.fromMessages([
  systemMessage,
  fewShotTemplate,
  ["human", "{input}"]
]);

// Implementation with chain
const chain = chatPrompt
  .pipe(new ChatOpenAI({ temperature: 0.7 }))
  .pipe(new StringOutputParser());

// Sample implementation
const response = await chain.invoke({
  input: "Where did Doc Brown acquire plutonium?"
});
```

## Dynamic Example Selection

```javascript
class BackToFutureExampleSelector {
  private examples: Array<{input: string, response: string}>;
  
  constructor(examples: Array<{input: string, response: string}>) {
    this.examples = examples;
  }

  async selectExamples(input: string): Promise<Array<{input: string, response: string}>> {
    // Choose the most pertinent examples based on the input
    // Could utilize embeddings, keyword matching, etc.
    return this.examples.slice(0, 2); // Basic example: return the first 2
  }
}

const dynamicPrompt = async (input: string) => {
  const selector = new BackToFutureExampleSelector(examples);
  const selectedExamples = await selector.selectExamples(input);
  
  return ChatPromptTemplate.fromMessages([
    systemMessage,
    FewShotChatMessagePromptTemplate.fromExamples(
      selectedExamples,
      {
        humanMessageTemplate: "{input}",
        aiMessageTemplate: "{response}"
      }
    ),
    ["human", input]
  ]);
};
```

## Best Practices

1. **Example Selection**
   - Opt for representative dialogues
   - Incorporate both affirmative and negative examples
   - Exhibit the desired tone and style

2. **Template Structure**
   - Initiate with a clear system message
   - Employ consistent formatting
   - Include examples handling errors

3. **Dynamic Selection**
   - Reflect on the relevance to the input
   - Cap the number of examples
   - Equilibrate the usage of the context window

# Example Selectors

## Overview
Example selectors facilitate the choice of suitable examples for few-shot prompts by employing selection strategies within the `select_examples` method.

The selection strategies comprise:
- Semantic similarity
- Maximal marginal relevance
- Length-based selection
- Random selection

## Basic Implementation

```javascript
import { BaseExampleSelector } from 'langchain/prompts';

class CustomExampleSelector extends BaseExampleSelector {
  private examples: Array<Record<string, string>>;

  constructor() {
    super();
    this.examples = [];
  }

  async addExample(example: Record<string, string>): Promise<void> {
    this.examples.push(example);
  }

  async selectExamples(input_variables: Record<string, string>): Promise<Array<Record<string, string>>> {
    // Basic random selection
    const numExamples = 2;
    const shuffled = [...this.examples].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, numExamples);
  }
}

// Usage Example
const selector = new CustomExampleSelector();

// Add examples
await selector.addExample({
  recipe: "Pasta Carbonara",
  instructions: "Cook pasta, mix with eggs, cheese, and pancetta"
});

await selector.addExample({
  recipe: "Chicken Curry",
  instructions: "Saute chicken with spices and coconut milk"
});

// Select random examples
const selectedExamples = await selector.selectExamples({});
```

## Integration with Few-Shot Templates

```javascript
import { FewShotPromptTemplate, PromptTemplate } from 'langchain/prompts';

const examplePrompt = PromptTemplate.fromTemplate(`
Recipe: {recipe}
Instructions: {instructions}
`);

const fewShotPrompt = new FewShotPromptTemplate({
  exampleSelector: selector,
  examplePrompt,
  prefix: "Here are some cooking recipes:",
  suffix: "Recipe: {input}\nInstructions:",
  inputVariables: ["input"]
});
```

## Best Practices

1. **Selector Design**
   - Contemplate input relevance
   - Execute efficient selection logic
   - Manage edge cases

2. **Example Management**
   - Sustain diverse example collections
   - Revise examples as required
   - Contemplate caching methods

3. **Performance**
   - Enhance selection algorithms
   - Contemplate batch processing
   - Observe selection duration

## Implementation Tips

- Commence with straightforward selection logic
- Test with a variety of input types
- Contemplate selection criteria
- Observe example diversity
- Execute proper error management

# Length-Based Example Selection

## Overview
Length-based example selection manages prompt length by selecting examples based on available context window space, ensuring prompts don't exceed model limitations.

## Basic Implementation

```javascript
import { 
  LengthBasedExampleSelector,
  FewShotPromptTemplate,
  PromptTemplate 
} from 'langchain/prompts';

// Example setup
const examples = [
  { 
    input: "I need coffee",
    output: "Sounds like you're feeling tired" 
  },
  { 
    input: "Just finished a marathon",
    output: "That's an amazing achievement!" 
  }
];

const examplePrompt = PromptTemplate.fromTemplate(`
User: {input}
Assistant: {output}
`);

// Initialize length-based selector
const selector = new LengthBasedExampleSelector({
  examples,
  examplePrompt,
  maxLength: 50 // Maximum length in tokens/characters
});

// Create dynamic prompt
const dynamicPrompt = new FewShotPromptTemplate({
  exampleSelector: selector,
  examplePrompt,
  prefix: "Respond to the user appropriately:",
  suffix: "User: {input}\nAssistant:",
  inputVariables: ["input"]
});

// Example usage with different input lengths
const longInput = await dynamicPrompt.format({
  input: "Here's a very long and detailed question that takes up most of our available context window..."
}); // May select fewer or no examples

const shortInput = await dynamicPrompt.format({
  input: "feeling wired"
}); // Can include more examples
```

## Selection Process

1. **Example Addition**
   - Format example with prompt
   - Calculate formatted length
   - Store in example text lengths

2. **Selection Logic**
   - Calculate input length
   - Determine remaining space
   - Add examples that fit
   - Track remaining length

## Best Practices

1. **Length Configuration**
   - Consider model token limits
   - Account for prefix/suffix
   - Leave room for response

2. **Example Management**
   - Order examples by priority
   - Maintain diverse lengths
   - Consider example relevance

## Implementation Tips

- Test with various input lengths
- Monitor selection behavior
- Consider token counting
- Balance quantity vs quality
- Handle edge cases

# Max Marginal Relevance (MMR) Example Selection

## Overview
MMR balances between example relevance and diversity by using cosine similarity calculations to avoid redundancy while maintaining relevance to the query.

Components:
- Relevance calculation (similarity to input)
- Diversity calculation (difference from selected examples)
- MMR score = λ * relevance - (1 - λ) * diversity

## Basic Implementation

```javascript
import { 
  MaxMarginalRelevanceExampleSelector,
  FewShotPromptTemplate,
  PromptTemplate 
} from 'langchain/prompts';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';

// Example setup
const examples = [
  { input: "happy", output: "sad" },
  { input: "tall", output: "short" },
  { input: "rich", output: "poor" },
  { input: "quiet", output: "loud" }
];

// Create MMR selector
const mmrSelector = await MaxMarginalRelevanceExampleSelector.fromExamples(
  examples,
  new OpenAIEmbeddings(),
  {
    fetchK: 5, // Number of examples to fetch before reranking
    k: 2,      // Number of examples to select
    lambda: 0.7 // Balance between relevance (1.0) and diversity (0.0)
  }
);

// Create prompt template
const examplePrompt = PromptTemplate.fromTemplate(`
Input: {input}
Output: {output}
`);

const dynamicPrompt = new FewShotPromptTemplate({
  exampleSelector: mmrSelector,
  examplePrompt,
  prefix: "What's the opposite of:",
  suffix: "Input: {adjective}\nOutput:",
  inputVariables: ["adjective"]
});

// Example usage
const result = await dynamicPrompt.format({
  adjective: "buzzed"
});
```

## Selection Process

1. **Initial Calculations**
   ```javascript
   // Pseudo-code for MMR selection
   function selectExamples(input, examples, k, lambda) {
     const selected = [];
     while (selected.length < k) {
       let bestScore = -Infinity;
       let bestExample = null;
       
       for (const example of examples) {
         const relevance = cosineSimilarity(input, example);
         const diversity = Math.max(
           ...selected.map(s => cosineSimilarity(s, example))
         );
         
         const mmrScore = lambda * relevance - (1 - lambda) * diversity;
         if (mmrScore > bestScore) {
           bestScore = mmrScore;
           bestExample = example;
         }
       }
       
       selected.push(bestExample);
     }
     return selected;
   }
   ```

## Best Practices

1. **Parameter Tuning**
   - Adjust lambda for relevance-diversity balance
   - Optimize fetchK and k values
   - Consider embedding model choice

2. **Performance Optimization**
   - Cache embeddings when possible
   - Pre-compute similarity matrices
   - Monitor computational overhead

## Implementation Tips

- Start with higher lambda values (~0.7)
- Experiment with different k values
- Consider task-specific requirements
- Monitor selection diversity
- Cache results when appropriate

# N-gram Overlap Example Selection

## Overview
N-gram Overlap Example Selector chooses examples based on their similarity to input using n-gram overlap scores. An n-gram is a contiguous sequence of n items from a text sample.

Example n-grams from "Sunsets are always beautiful":
- Unigrams: ["Sunsets", "are", "always", "beautiful"]
- Bigrams: ["Sunsets are", "are always", "always beautiful"]
- Trigrams: ["Sunsets are always", "are always beautiful"]

## Basic Implementation

```javascript
import { 
  NGramOverlapExampleSelector,
  FewShotPromptTemplate,
  PromptTemplate 
} from 'langchain/prompts';

// Example setup
const examples = [
  { input: "The cat sits", output: "Simple present tense" },
  { input: "The dog ran", output: "Simple past tense" },
  { input: "Birds are flying", output: "Present continuous" }
];

const examplePrompt = PromptTemplate.fromTemplate(`
Input: {input}
Output: {output}
`);

// Create selector with threshold
const ngramSelector = new NGramOverlapExampleSelector({
  examples,
  examplePrompt,
  threshold: -1.0  // Default: include all, just reorder
});

// Create prompt template
const dynamicPrompt = new FewShotPromptTemplate({
  exampleSelector: ngramSelector,
  examplePrompt,
  prefix: "Identify the tense in each sentence:",
  suffix: "Input: {sentence}\nOutput:",
  inputVariables: ["sentence"]
});

// Example usage with different thresholds
const examples1 = await dynamicPrompt.format({
  sentence: "The cat is sleeping"
}); // All examples, reordered by similarity

const ngramSelectorStrict = new NGramOverlapExampleSelector({
  examples,
  examplePrompt,
  threshold: 0.25  // Only include examples with significant overlap
});

const examples2 = await dynamicPrompt.format({
  sentence: "The cat is sleeping"
}); // Fewer, more relevant examples
```

## Threshold Values and Behavior

1. **Threshold = -1.0 (Default)**
   - Include all examples
   - Reorder based on overlap score

2. **Threshold = 0.0**
   - Include examples with any overlap
   - Exclude examples with no overlap

3. **Threshold > 0.0**
   - Include examples above threshold
   - More selective with higher values

4. **Threshold > 1.0**
   - Excludes all examples
   - Returns empty list

## Best Practices

1. **Threshold Selection**
   - Start with default (-1.0)
   - Adjust based on example quality
   - Monitor example count

2. **Example Management**
   - Ensure diverse n-gram patterns
   - Consider sentence structure
   - Balance example complexity

## Implementation Tips

- Test different threshold values
- Consider sentence structure
- Monitor selection behavior
- Balance precision vs. recall
- Cache overlap calculations

# Semantic Similarity Example Selection

## Overview
The Semantic Similarity Example Selector uses embeddings and vector stores to find examples that are semantically most similar to the input query.

## Basic Implementation

```javascript
import { 
  SemanticSimilarityExampleSelector,
  FewShotPromptTemplate,
  PromptTemplate 
} from 'langchain/prompts';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { FaissStore } from 'langchain/vectorstores/faiss';

// Example setup
const examples = [
  { input: "The weather is great today", output: "Positive sentiment" },
  { input: "This movie was terrible", output: "Negative sentiment" },
  { input: "I feel neutral about this", output: "Neutral sentiment" }
];

// Create selector
const semanticSelector = await SemanticSimilarityExampleSelector.fromExamples(
  examples,
  new OpenAIEmbeddings(),
  FaissStore,
  {
    k: 4,  // Number of examples to select
    exampleKeys: ["input", "output"],  // Optional: filter by specific keys
    inputKeys: ["input"]  // Optional: consider only specific input keys
  }
);

// Create prompt template
const examplePrompt = PromptTemplate.fromTemplate(`
Example Input: {input}
Example Output: {output}
`);

const dynamicPrompt = new FewShotPromptTemplate({
  exampleSelector: semanticSelector,
  examplePrompt,
  prefix: "Classify the sentiment of each sentence:",
  suffix: "Input: {sentence}\nOutput:",
  inputVariables: ["sentence"]
});

// Chain usage
const chain = dynamicPrompt
  .pipe(new ChatOpenAI())
  .pipe(new StringOutputParser());

const response = await chain.invoke({
  sentence: "I absolutely love this product!"
});
```

## Selection Process

1. **Initialization**
   - Create embeddings for all examples
   - Store in vector store
   - Configure selection parameters

2. **Selection Logic**
   - Embed input query
   - Find nearest neighbors
   - Filter by keys (if specified)
   - Return top-k examples

## Best Practices

1. **Embedding Selection**
   - Choose appropriate embedding model
   - Consider domain specificity
   - Balance quality vs. performance

2. **Vector Store Configuration**
   - Choose appropriate store type
   - Consider indexing strategy
   - Optimize for query speed

3. **Example Management**
   - Maintain diverse examples
   - Update embeddings as needed
   - Monitor selection quality

## Implementation Tips

- Cache embeddings for efficiency
- Monitor similarity thresholds
- Consider batch processing
- Test with domain-specific content
- Validate semantic relevance

# Partial Prompt Templates

## Overview
Partial prompt templates allow predefinition of some template variables while leaving others for later. Two main methods:
- Partial with strings: Pre-fill known string values
- Partial with functions: Use functions for dynamic values

## Basic Implementation

### Partial with Strings
```javascript
import { PromptTemplate } from 'langchain/prompts';

// Template with multiple variables
const template = new PromptTemplate({
  template: "Hello {username}, hope you're enjoying your {user_activity} at {location}",
  inputVariables: ["username", "user_activity", "location"]
});

// Create partial template with preset username
const partialTemplate = template.partial({
  username: "Alice"
});

// Use partial template later
const response = await partialTemplate.format({
  user_activity: "time",
  location: "Wonderland Resort"
});
// Output: "Hello Alice, hope you're enjoying your time at Wonderland Resort"
```

### Partial with Functions
```javascript
const getWeather = () => "sunny"; // Simplified example

const greetingTemplate = new PromptTemplate({
  template: "Good morning! It's {weather} today. Would you like to {activity}?",
  inputVariables: ["weather", "activity"]
});

const weatherTemplate = greetingTemplate.partial({
  weather: getWeather
});

const response = await weatherTemplate.format({
  activity: "go for a walk"
});
// Output: "Good morning! It's sunny today. Would you like to go for a walk?"
```

## Best Practices

1. **Variable Management**
   - Identify static vs. dynamic variables
   - Group related variables
   - Document dependencies

2. **Function Implementation**
   - Keep functions pure
   - Handle errors gracefully
   - Consider caching results

3. **Template Organization**
   - Modularize common patterns
   - Version template variations
   - Document partial requirements

## Implementation Tips

- Pre-fill commonly used values
- Use functions for real-time data
- Consider error handling
- Cache function results
- Document partial variables

# Chain of Thought Prompting

## Overview
Chain of Thought (CoT) prompting, introduced by Wei et al. (2022), enhances language model reasoning by guiding them through intermediate steps before reaching conclusions.

Key benefits:
- Breaks down complex problems
- Improves reasoning capabilities
- Provides interpretable responses
- Works well with few-shot prompting

## Basic Implementation

```javascript
import { 
  MaxMarginalRelevanceExampleSelector,
  FewShotPromptTemplate,
  PromptTemplate 
} from 'langchain/prompts';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { FaissStore } from 'langchain/vectorstores/faiss';

// Example setup with reasoning steps
const examples = [
  {
    question: "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many does he have now?",
    reasoning: `Let's solve this step by step:
1. Initial balls = 5
2. New cans = 2
3. Balls per can = 3
4. Additional balls = 2 cans × 3 balls = 6 balls
5. Total balls = Initial + Additional = 5 + 6 = 11`,
    answer: "11 tennis balls"
  }
];

// Create prompt template
const examplePrompt = PromptTemplate.fromTemplate(`
Question: {question}
Reasoning: {reasoning}
Answer: {answer}
`);

// Setup MMR selector
const selector = await MaxMarginalRelevanceExampleSelector.fromExamples(
  examples,
  new OpenAIEmbeddings(),
  FaissStore,
  {
    k: 2,
    fetchK: 3
  }
);

// Create chain of thought prompt
const cotPrompt = new FewShotPromptTemplate({
  exampleSelector: selector,
  examplePrompt,
  prefix: "Consider the following examples of step-by-step reasoning:",
  suffix: `
Now solve this question step by step:
Question: {input}
Reasoning:`,
  inputVariables: ["input"]
});

// Usage with LLM
const chain = cotPrompt
  .pipe(new ChatOpenAI({ temperature: 0 }))
  .pipe(new StringOutputParser());

const response = await chain.invoke({
  input: "If a chest contracts, and we know inhaling increases chest size and decreases lung pressure, what happens to lung pressure when the chest contracts?"
});
```

## Best Practices

1. **Example Selection**
   - Use clear, step-by-step reasoning
   - Include diverse problem types
   - Match complexity to task

2. **Prompt Structure**
   - Start with clear examples
   - Include explicit reasoning steps
   - End with clear output format

3. **Implementation Tips**
   - Use zero temperature for consistent reasoning
   - Include "take a deep breath" type instructions
   - Consider adding helpful tips in prompts

## Use Cases

- Complex arithmetic problems
- Logic puzzles
- Common sense reasoning
- Scientific explanations
- Process analysis

## Implementation Tips

- Break down complex problems
- Show intermediate steps
- Validate reasoning chains
- Use with few-shot learning
- Monitor reasoning quality

# Chain of Thought Prompting

## Overview
Chain of Thought (CoT) prompting, introduced by Wei et al. (2022), guides language models through intermediate reasoning steps for complex tasks. Rather than direct answers, it encourages step-by-step thinking.

## Example Problem Solving

```javascript
const mathProblem = {
  question: "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many does he have now?",
  reasoning: `
Let's solve this step by step:
1. Initial count: Roger starts with 5 balls
2. New purchases: 2 cans × 3 balls per can = 6 new balls
3. Final count: 5 initial + 6 new = 11 balls total
  `,
  answer: "11 tennis balls"
};
```

## Vector Store Integration

```javascript
import { MaxMarginalRelevanceExampleSelector } from 'langchain/prompts';
import { HuggingFaceBgeEmbeddings } from 'langchain/embeddings';
import { FaissStore } from 'langchain/vectorstores/faiss';

// Setup embeddings and store
const embeddings = new HuggingFaceBgeEmbeddings();
const examples = await loadExamplesFromHuggingFace(); // Your loading function

const selector = await MaxMarginalRelevanceExampleSelector.fromExamples(
  examples,
  embeddings,
  FaissStore,
  { k: 4, fetchK: 20 }
);

const examplePrompt = PromptTemplate.fromTemplate(`
Question: {question}
Reasoning: {reasoning}
Answer: {answer}
`);

const chainOfThoughtPrompt = new FewShotPromptTemplate({
  exampleSelector: selector,
  examplePrompt,
  prefix: "Consider the following examples of step-by-step reasoning:",
  suffix: `
Using a similar reasoning approach, answer the user's question.
Take a deep breath and break down the query step-by-step.

Question: {input}
Reasoning:`,
  inputVariables: ["input"]
});
```

## Best Practices

1. **Prompt Structure**
   - Include clear reasoning steps
   - Break down complex problems
   - Show intermediate calculations
   - End with conclusive answers

2. **Example Selection**
   - Use relevant examples from vector store
   - Maintain diverse reasoning patterns
   - Include edge cases

3. **Implementation Tips**
   - Set temperature to 0 for consistent reasoning
   - Use "take a deep breath" type instructions
   - Consider adding helpful hints in prompts

## Use Cases
- Multi-step arithmetic problems
- Logical reasoning tasks
- Scientific problem solving
- Common sense reasoning
- Process analysis

# Self-Consistency Prompting

## Overview
Self-consistency prompting (Wang et al., 2022) explores multiple valid reasoning paths to achieve more reliable answers. Unlike standard chain-of-thought, it samples multiple paths and determines the most consistent answer among them.

## Implementation Example

```javascript
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PromptTemplate } from 'langchain/prompts';

// Setup for multiple reasoning paths
const selfConsistencyTemplate = `
Based on the responses delimited by [brackets] to the following query delimited by triple backticks, return the response that occurs most frequently.

Query: ```{query}```

Let's solve this step by step through multiple approaches:

[Response 1]
{reasoning1}

[Response 2]
{reasoning2}

[Response 3]
{reasoning3}

Most consistent answer:`;

// Generate multiple reasoning paths
async function generateMultipleReasonings(query, n = 5) {
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    n: n
  });

  const cotPrompt = chainOfThoughtPrompt.format({
    input: query
  });

  const generations = await model.generate([cotPrompt], n);
  return generations.generations.map(g => g.text);
}

// Find most consistent answer
async function findMostConsistentAnswer(query) {
  const reasonings = await generateMultipleReasonings(query);
  
  const selfConsistencyPrompt = PromptTemplate.fromTemplate(selfConsistencyTemplate);
  const finalPrompt = await selfConsistencyPrompt.format({
    query,
    reasoning1: reasonings[0],
    reasoning2: reasonings[1],
    reasoning3: reasonings[2]
  });

  const model = new ChatOpenAI({ temperature: 0 });
  return await model.call(finalPrompt);
}

// Usage example
const response = await findMostConsistentAnswer(
  "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are there?"
);
```

## Best Practices

1. **Multiple Paths Generation**
   - Use temperature > 0 for diversity
   - Generate enough paths (3-5 minimum)
   - Maintain reasoning structure

2. **Answer Aggregation**
   - Consider frequency of answers
   - Look for reasoning consistency
   - Handle contradictions

3. **Implementation Tips**
   - Use structured prompts
   - Balance path diversity
   - Consider computational costs

## Use Cases
- Complex arithmetic problems
- Multi-step reasoning tasks
- Ambiguous scenarios
- Verification of solutions
- Quality assurance

# Self-Ask Prompting

## Overview
Self-Ask prompting (Press et al., 2022) addresses the compositional reasoning gap in language models by enabling them to break down complex questions into smaller, manageable sub-questions.

## Implementation Example

```javascript
import { 
  ChatOpenAI,
  AgentExecutor,
  DuckDuckGoSearchTools,
  SelfAskOutputParser
} from 'langchain';

// Setup tools
const search = new DuckDuckGoSearchTools();
const tools = [{
  name: "intermediate_answer",
  func: search.run.bind(search),
  description: "Search engine to find answers to factual questions"
}];

// Create self-ask prompt
const selfAskTemplate = `Question: {question}

Let's approach this step-by-step:

1) First, let me check if I need to search for any information.
2) If yes, I'll ask a follow-up question and search for the answer.
3) I'll repeat this process until I have enough information.
4) Finally, I'll provide a complete answer.

Follow-up: Do I need to search for any information to answer this question?

{agent_scratchpad}`;

// Create agent with stop sequence
const model = new ChatOpenAI({ temperature: 0 })
  .bind({ stop: ["intermediate answer"] });

// Create self-ask agent
const agent = await createSelfAskWithSearch({
  llm: model,
  tools,
  prompt: selfAskTemplate
});

// Execute agent
const result = await AgentExecutor.fromAgentAndTools({
  agent,
  tools,
  verbose: true
}).invoke({
  question: "What is the RWKV architecture for LLMs?"
});
```

## Modern Implementation (Recommended)

```javascript
import { createSelfAskWithSearchAgent } from 'langchain/agents';

const agent = await createSelfAskWithSearchAgent({
  llm: model,
  tools,
  prompt: selfAskTemplate
});

const executor = new AgentExecutor({
  agent,
  tools,
  verbose: true
});

const response = await executor.invoke({
  input: "What is Direct Preference Optimization?"
});
```

## Best Practices

1. **Question Decomposition**
   - Break complex queries into sub-questions
   - Validate intermediate answers
   - Monitor reasoning chains

2. **Search Integration**
   - Use reliable search tools
   - Validate search results
   - Handle search failures gracefully

3. **Implementation Tips**
   - Set appropriate stop sequences
   - Monitor for hallucinations
   - Implement error handling
   - Consider rate limits

## Use Cases
- Complex factual queries
- Multi-step reasoning tasks
- Research questions
- Fact verification
- Knowledge exploration

# Core Concepts

## Prompting Strategies

### Direct Prompting
The simplest form of interaction:
- Single question/instruction
- Direct response
- No intermediate steps
- Limited reasoning capability

### Chain of Thought (CoT)
Encourages step-by-step reasoning:
- Breaks down complex problems
- Shows intermediate steps
- Improves accuracy
- Better for mathematical/logical tasks

### Zero-Shot Prompting
Handling new tasks without examples:
- No training examples needed
- Uses model's existing knowledge
- Often enhanced with phrases like "Let's think step by step"
- Lower accuracy than few-shot methods

### Few-Shot Prompting
Learning from examples in the prompt:
- Include 2-5 examples
- Demonstrates desired format
- Improves consistency
- Better for specific formats/styles

## Advanced Techniques

### Retrieval Augmentation
Enhancing responses with external knowledge:
- Real-time information access
- Reduced hallucination
- Dynamic knowledge updates
- Source verification

### Active Retrieval
Dynamic information gathering:
- Confidence-based retrieval
- Self-questioning
- Iterative refinement
- Adaptive context building

### Planning and Execution
Structured problem-solving approach:
- Explicit planning phase
- Subtask breakdown
- Sequential execution
- Progress monitoring

## Practical Considerations

### Model Selection
Factors to consider:
- Context window size
- Token costs
- Temperature settings
- Model capabilities

### Error Handling
Common challenges:
- Token limits
- Rate limiting
- Error recovery
- Fallback strategies

### Performance Optimization
Key strategies:
- Caching
- Batching
- Prompt optimization
- Response streaming

# ReAct Prompting

## Overview
ReAct (Reasoning and Acting) framework, introduced by Google researchers in 2022, combines reasoning traces with task-specific actions in language models. It enables:
- Dynamic plan updates
- Exception handling
- External information gathering

## Comparison with Other Methods
1. Standard Prompting: Direct question → answer
2. Chain of Thought: Reasoning examples → guided thinking
3. Action Only: Action examples → guided actions
4. ReAct: Interleaved reasoning and actions

## Implementation Example

```javascript
import { 
  DuckDuckGoSearchTools,
  WikipediaQueryRun,
  createReactAgent,
  AgentExecutor 
} from 'langchain';

// Setup tools
const searchTool = new DuckDuckGoSearchTools();
const wikiTool = new WikipediaQueryRun();

const tools = [
  {
    name: "search",
    func: searchTool.run.bind(searchTool),
    description: "Search the web for current information"
  },
  {
    name: "wikipedia",
    func: wikiTool.run.bind(wikiTool),
    description: "Search Wikipedia articles"
  }
];

// Create ReAct agent
const model = new ChatOpenAI({ modelName: "gpt-4", temperature: 0 });

const agent = await createReactAgent({
  llm: model,
  tools,
  verbose: true
});

const executor = new AgentExecutor({
  agent,
  tools,
  verbose: true
});

// Example usage
const response = await executor.invoke({
  input: "Who is playing in the Super Bowl 2024? Where will it be played?"
});
```

## Tool Integration

ReAct supports various tools including:
- Web search (DuckDuckGo, Bing)
- Knowledge bases (Wikipedia)
- Archives
- Custom tools

## Best Practices

1. **Tool Selection**
   - Choose appropriate tools for task
   - Provide clear tool descriptions
   - Consider tool reliability

2. **Reasoning Process**
   - Monitor reasoning traces
   - Validate external information
   - Handle tool failures

3. **Implementation Tips**
   - Use verbose mode for debugging
   - Implement error handling
   - Consider rate limits
   - Monitor API usage

## Use Cases
- Question answering
- Fact verification
- Web navigation
- Research tasks
- Document analysis

# Retrieval Augmented Generation (RAG)

## Overview
Introduced by Meta AI in 2020, RAG enhances LLMs by providing access to external knowledge sources, addressing limitations like:
- Expensive training/fine-tuning
- Static knowledge bases
- Hallucination problems

## Core Components

1. **Language Model**
   - Generates human-like text responses
   - Processes augmented context

2. **Vector Store**
   - Houses embeddings and text data
   - Enables efficient similarity search

3. **Document Processing**
   - Loaders: Import from various sources
   - Chunkers: Break documents into manageable pieces
   - Embedders: Convert text to vector representations

## Implementation Example

```javascript
import { 
  RecursiveCharacterTextSplitter,
  OpenAIEmbeddings,
  FaissStore,
  ChatOpenAI,
  loadPrompt 
} from 'langchain';

// Document Loading and Processing
const webLoader = new WebBaseLoader([
  'https://example.com/article1',
  'https://example.com/article2'
]);
const docs = await webLoader.load();

// Text Splitting
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50
});
const chunks = await textSplitter.splitDocuments(docs);

// Embeddings and Storage
const embeddings = new CacheBackedEmbeddings({
  underlyingEmbeddings: new OpenAIEmbeddings(),
  documentEmbeddings: new InMemoryStore()
});

const vectorStore = await FaissStore.fromDocuments(
  chunks,
  embeddings
);

// Create Retriever
const retriever = vectorStore.asRetriever();

// Setup RAG Chain
const formatDocs = (docs) => docs.map(doc => doc.pageContent).join('\n\n');

const ragChain = RunnableSequence.from([
  {
    context: retriever.pipe(formatDocs),
    question: new RunnablePassthrough()
  },
  await loadPrompt("rag/chat"),
  new ChatOpenAI({ modelName: "gpt-4" }),
  new StringOutputParser()
]);

// Example Usage
const response = await ragChain.invoke(
  "What is Neural Architecture Search?"
);
```

## Index Subsystem Steps

1. **Document Loading**
   - Import from various sources
   - Convert to document objects

2. **Text Processing**
   - Split into chunks
   - Create embeddings
   - Store in vector database

3. **Retrieval System**
   - Process user query
   - Find similar documents
   - Return relevant matches

## Best Practices

1. **Document Processing**
   - Choose appropriate chunk sizes
   - Consider content overlap
   - Monitor embedding quality

2. **Vector Store Management**
   - Use caching when possible
   - Monitor storage efficiency
   - Consider scaling requirements

3. **Query Processing**
   - Implement error handling
   - Monitor retrieval quality
   - Balance speed vs accuracy

## Implementation Tips

- Use cached embeddings
- Monitor token usage
- Consider batch processing
- Implement logging
- Handle edge cases

# Forward-Looking Active REtrieval (FLARE)

## Overview
Introduced in 2023, FLARE enhances RAG by actively retrieving information during generation. Unlike traditional RAG that retrieves once upfront, FLARE continuously checks confidence and retrieves as needed.

Key features:
- Active retrieval during generation
- Confidence-based token checking
- Forward-looking query generation
- Iterative refinement

## Implementation Example

```javascript
import { 
  OpenAI,
  ChatOpenAI,
  FLAREChain,
  ArchiveRetriever 
} from 'langchain';

// Setup components
const llm = new OpenAI({
  modelName: "text-davinci-003",
  temperature: 0,
  streaming: true,
  logprobs: true
});

const questionGenerator = new ChatOpenAI({
  temperature: 0
});

// Create retriever
const retriever = vectorStore.asRetriever();

// Initialize FLARE chain
const flareChain = await FLAREChain.fromLLM({
  llm,
  retriever,
  questionGeneratorLLM: questionGenerator,
  maxGenerationLength: 128,
  minProbability: 0.2,
  verbose: true,
  handleParsingErrors: true
});

// Example usage
const response = await flareChain.invoke({
  query: "Explain the steps in the FLARE pipeline"
});
```

## Process Steps

1. **Initial Generation**
   - Start with user input
   - Retrieve initial context
   - Begin generating response

2. **Confidence Checking**
   - Monitor token probabilities
   - Identify uncertain segments
   - Generate clarifying questions

3. **Active Retrieval**
   - Use generated questions
   - Fetch relevant documents
   - Update generation context

4. **Iterative Refinement**
   - Regenerate uncertain parts
   - Continue until confident
   - Maintain coherence

## Best Practices

1. **Model Selection**
   - Use models with logprobs
   - Balance speed vs accuracy
   - Consider token limits

2. **Parameter Tuning**
   - Adjust confidence thresholds
   - Optimize generation length
   - Configure retrieval frequency

3. **Performance Optimization**
   - Cache retrievals
   - Monitor generation time
   - Handle timeouts

## Implementation Tips

- Use OpenAI for logprobs
- Monitor token confidence
- Cache intermediate results
- Handle long generations
- Implement timeout logic

# Plan and Execute Prompting

## Overview
Plan-and-Solve prompting (Wang et al., 2023) enhances zero-shot chain-of-thought reasoning by dividing complex tasks into planning and execution phases.

Key differences from zero-shot chain-of-thought:
- Explicit planning phase
- Structured execution
- Enhanced error handling (Plan-and-Solve Plus)

## Implementation Example

```javascript
import { 
  PlanAndExecuteAgent,
  initializeAgentExecutorWithOptions,
  DuckDuckGoSearchTools,
  Calculator 
} from 'langchain/experimental';

// Setup components
const model = new ChatOpenAI({
  modelName: "gpt-4-0125-preview",
  temperature: 0
});

// Setup tools
const tools = [
  new DuckDuckGoSearchTools(),
  new Calculator()
];

// Create planner
const planner = await initializeAgentExecutorWithOptions(
  tools,
  model,
  {
    agentType: "plan-and-execute",
    verbose: true
  }
);

// Create executor
const executor = await initializeAgentExecutorWithOptions(
  tools,
  model,
  {
    agentType: "structured-chat-zero-shot-react-description",
    verbose: true
  }
);

// Initialize agent
const agent = new PlanAndExecuteAgent({
  planner,
  executor,
  verbose: true
});

// Example usage
const response = await agent.invoke({
  input: "What's the current temperature in Winnipeg? Get the absolute value."
});
```

## Process Steps

1. **Planning Phase**
   ```javascript
   const plannerPrompt = `
   Let's understand the problem and devise a plan.
   Please output the plan starting with 'Plan:' followed by numbered steps.
   
   Current goal: {input}
   Available tools: {tools}
   `;
   ```

2. **Execution Phase**
   ```javascript
   const executorPrompt = `
   Execute the following plan step by step:
   {plan}
   
   Current step: {step}
   Previous steps completed: {previous_steps}
   `;
   ```

## Best Practices

1. **Agent Configuration**
   - Use temperature = 0
   - Enable verbose mode
   - Consider tool selection

2. **Planning Strategy**
   - Break tasks into subtasks
   - Handle dependencies
   - Consider edge cases

3. **Execution Optimization**
   - Monitor step completion
   - Handle failures gracefully
   - Track progress

## Use Cases
- Complex multi-step tasks
- Tasks requiring coordination
- Research and analysis
- Data gathering and processing
- Sequential decision making

# Prompt Management

## Overview
Prompt engineering requires systematic version control and management due to LLMs' high sensitivity to prompt changes. Even minor modifications in:
- Wording
- Punctuation
- Spacing
- Line breaks
can significantly impact model outputs.

## Challenge Areas

### Version Control
```javascript
// Example of prompt versioning
const promptVersions = {
  v1: "Summarize this text",
  v2: "Summarize this text briefly",
  v3: "Summarize this text briefly and clearly",
  current: "Provide a brief, clear summary of this text"
};
```

### Change Tracking
```javascript
const promptChangelog = [
  {
    version: "1.0.0",
    date: "2024-02-01",
    changes: "Initial prompt structure",
    prompt: "Summarize text"
  },
  {
    version: "1.0.1",
    date: "2024-02-02",
    changes: "Added clarity instruction",
    prompt: "Summarize text clearly"
  }
];
```

## LangSmith Integration

### Basic Setup
```javascript
import { Client } from "langsmith";

const client = new Client({
  apiKey: process.env.LANGSMITH_API_KEY
});

// Create project
const project = await client.createProject({
  name: "prompt-optimization",
  reference_id: "prompt-opt-001"
});

// Track prompt runs
const run = await client.createRun({
  name: "summarization-prompt",
  project_id: project.id,
  inputs: { text: "Sample text" },
  runtime: {
    prompt_template: promptVersions.current
  }
});
```

### Prompt Evaluation
```javascript
// Evaluate prompt performance
const evaluation = await client.createEvaluation({
  run_id: run.id,
  metrics: {
    relevance: 0.85,
    clarity: 0.9,
    accuracy: 0.88
  }
});
```

# LangSmith Platform

## Overview
LangSmith is a unified platform for debugging, testing, and monitoring LLM applications. It bridges the gap between prototyping and production deployment by providing:
- Comprehensive debugging tools
- Testing frameworks
- Performance monitoring
- Usage analytics

## Core Features

### Development Support
```javascript
// Development environment setup
const developmentConfig = {
  langsmith: {
    apiKey: process.env.LANGSMITH_API_KEY,
    projectName: "development",
    traceOptions: {
      enableDebug: true,
      captureInputs: true,
      captureOutputs: true
    }
  }
};
```

### Testing Framework
```javascript
// Example test suite setup
const testSuite = {
  name: "Chain Validation",
  tests: [
    {
      description: "Input validation test",
      input: { query: "test query" },
      expectedOutput: { type: "string", minLength: 10 },
      evaluationMetrics: ["accuracy", "latency"]
    }
  ],
  runConfig: {
    batchSize: 10,
    parallel: true
  }
};
```

### Monitoring System
```javascript
// Production monitoring setup
const monitoringSetup = {
  metrics: ["latency", "tokenUsage", "errorRate"],
  alerts: {
    errorThreshold: 0.05,
    latencyThreshold: 2000,
    notification: {
      email: "alerts@example.com",
      slack: "#monitoring"
    }
  }
};
```

## Use Cases

1. **Development Phase**
   - Component debugging
   - Prompt optimization
   - Performance profiling
   - Error tracing

2. **Testing Phase**
   - Automated testing
   - Quality assurance
   - Regression testing
   - Performance benchmarking

3. **Production Phase**
   - Live monitoring
   - Usage analytics
   - Cost tracking
   - Performance optimization

## Integration Example

```javascript
import { Client } from "langsmith";

// Initialize LangSmith client
const langsmith = new Client({
  apiKey: process.env.LANGSMITH_API_KEY,
  projectName: "production-app"
});

// Create trace
const trace = await langsmith.createTrace({
  name: "user-query-processing",
  metadata: {
    environment: "production",
    version: "1.0.0"
  }
});

// Track chain execution
const trackChain = async (chain, input) => {
  const run = await langsmith.createRun({
    name: chain.name,
    traceId: trace.id,
    inputs: input,
    startTime: new Date()
  });

  try {
    const result = await chain.invoke(input);
    await langsmith.updateRun({
      runId: run.id,
      outputs: result,
      endTime: new Date(),
      status: "completed"
    });
    return result;
  } catch (error) {
    await langsmith.updateRun({
      runId: run.id,
      error: error.message,
      endTime: new Date(),
      status: "failed"
    });
    throw error;
  }
};
```

## Best Practices

1. **Development Best Practices**
   - Enable detailed tracing
   - Use meaningful run names
   - Tag environments
   - Track metadata

2. **Testing Best Practices**
   - Create comprehensive test suites
   - Use realistic test data
   - Monitor performance metrics
   - Track regression issues

3. **Production Best Practices**
   - Set up alerting
   - Monitor costs
   - Track usage patterns
   - Analyze performance trends

## Deployment Considerations

1. **Security**
   - API key management
   - Data privacy
   - Access control
   - Audit logging

2. **Scaling**
   - Resource management
   - Performance optimization
   - Cost monitoring
   - Usage quotas

# LangSmith Walkthrough

## Initial Setup

1. **Registration**
   - Visit smith.langchain.com
   - Sign up options:
     - Discord
     - GitHub
     - Google
     - Email

2. **API Key Management**
   ```javascript
   // Store API key securely in environment variables
   const config = {
     langsmith: {
       apiKey: process.env.LANGSMITH_API_KEY,
       baseUrl: "https://api.smith.langchain.com"
     }
   };
   ```

## UI Components

### Projects Dashboard
- Run history
- Execution monitoring
- Performance metrics
- Debug information

### Annotation Interface
```javascript
// Example annotation setup
const annotationConfig = {
  queue: "code-review",
  metadata: {
    reviewer: "expert-dev",
    criteria: ["correctness", "efficiency", "style"]
  },
  feedback: {
    type: "structured",
    options: ["approved", "needs_revision", "rejected"]
  }
};
```

### Hub Management

1. **Creating New Prompts**
   ```javascript
   // Example prompt creation
   const newPrompt = {
     name: "linkedin-learning-example",
     description: "Prompt for LangSmith section of LinkedIn Learning course",
     useCase: "code-writing",
     language: "English",
     model: "gpt-3.5-turbo",
     isPrivate: true,
     template: `
       You're an expert in PyTorch, Hugging Face, and OpenAI.
       You're helping a user write code to complete their work.
       Be brief and speak as if you were a punk rocker.

       Question: {query}
     `
   };
   ```

2. **Prompt Settings**
   ```javascript
   const promptSettings = {
     model: "gpt-3.5-turbo",
     temperature: 0.2,
     maxTokens: 1000,
     topP: 0.9,
     frequencyPenalty: 0,
     presencePenalty: 0
   };
   ```

## Best Practices

1. **API Key Security**
   - Never expose keys in code
   - Use environment variables
   - Rotate keys regularly
   - Monitor key usage

2. **Prompt Version Control**
   - Use descriptive names
   - Add detailed descriptions
   - Document use cases
   - Track modifications

3. **Testing Strategy**
   - Test with various inputs
   - Monitor response quality
   - Track performance metrics
   - Document edge cases

## Implementation Example

```javascript
import { LangSmithClient } from "langsmith";

// Initialize client with secure configuration
const client = new LangSmithClient({
  apiKey: process.env.LANGSMITH_API_KEY
});

// Create and test a prompt
const testPrompt = async () => {
  // Create prompt
  const prompt = await client.createPrompt({
    ...newPrompt,
    settings: promptSettings
  });

  // Test prompt
  const result = await client.testPrompt(prompt.id, {
    query: "Write a PyTorch training loop"
  });

  // Save version if successful
  if (result.success) {
    await client.createPromptVersion(prompt.id, {
      version: "1.0.0",
      changes: "Initial working version"
    });
  }

  return result;
};
```

# Prompt Versioning in LangSmith

## Overview
Prompt versioning ensures stability in production environments by allowing specific version usage and tracking changes. LangChain Hub supports version control through commit hashes.

## Implementation

### Loading Versioned Prompts
```javascript
import { loadPrompt } from 'langchain/prompts/load';

// Load specific version of a prompt
const loadVersionedPrompt = async () => {
  const promptId = "datascienceharp/linkedin-learning-example@v1";
  return await loadPrompt(promptId);
};

// Using with LLM
const chain = prompt
  .pipe(new ChatOpenAI())
  .pipe(new StringOutputParser());
```

### Prompt Management
```javascript
// Example prompt modification
const updatePrompt = async (originalPrompt) => {
  // Add new instructions
  originalPrompt += "\nWrite code that follows PEP8 standards";
  originalPrompt += "\nYou also hate being nice";
  
  return originalPrompt;
};

// Push to LangChain Hub
const pushPrompt = async (prompt, accountHandle, promptName) => {
  await client.pushPrompt(`${accountHandle}/${promptName}`, prompt);
};
```

## Version Control Features

1. **Commit Management**
   ```javascript
   const versionControl = {
     getCommitHistory: async (promptId) => {
       const history = await client.getPromptHistory(promptId);
       return history.map(commit => ({
         hash: commit.hash,
         timestamp: commit.timestamp,
         changes: commit.message
       }));
     },
     
     getSpecificVersion: async (promptId, version) => {
       return await client.getPromptVersion(`${promptId}@${version}`);
     }
   };
   ```

2. **Prompt Updates**
   ```javascript
   const promptUpdate = async (promptId, newContent) => {
     const prompt = await client.getPrompt(promptId);
     prompt.content = newContent;
     
     // Create new version
     const version = await client.createVersion(promptId, {
       content: prompt.content,
       message: "Updated prompt content"
     });
     
     return version;
   };
   ```

## Best Practices

1. **Version Management**
   - Use specific versions in production
   - Test changes before deployment
   - Document version differences
   - Maintain changelog

2. **Deployment Strategy**
   - Stage changes in development
   - Test thoroughly
   - Roll out gradually
   - Monitor performance

3. **Collaboration**
   - Use descriptive commit messages
   - Review changes
   - Document dependencies
   - Track issues
