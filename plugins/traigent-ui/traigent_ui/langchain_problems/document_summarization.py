"""
Document Summarization Quality Problem.

A challenging summarization problem that tests the model's ability to:
1. Extract key information while maintaining accuracy
2. Balance brevity with completeness
3. Maintain coherence and readability
4. Adapt summary length to content complexity
"""

import sys
from typing import Any, Callable, Dict, List, Optional

import traigent
from traigent.evaluators.base import Dataset, EvaluationExample

try:
    from langchain.chains import LLMChain
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:
    print("Please install LangChain: pip install langchain langchain-openai")
    sys.exit(1)

from . import register_problem
from .base import BaseLangChainProblem, ProblemConfig, ProblemMetric


class DocumentSummarizationProblem(BaseLangChainProblem):
    """
    Document summarization quality problem.

    Tests the model's ability to create high-quality summaries of various document types
    while balancing informativeness, accuracy, and conciseness.
    """

    @classmethod
    def get_default_config(cls) -> ProblemConfig:
        """Get default configuration for this problem."""
        return ProblemConfig(
            name="document_summarization",
            description="High-quality document summarization with factual accuracy and coherence",
            difficulty_level="Advanced",
            dataset_size=20,
            model_configurations={
                "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
                "temperature": [0.3, 0.7],
                "max_tokens": [150, 300],
            },
            metrics=[
                ProblemMetric(
                    "content_coverage", "Coverage of key content", True, 1.0, ".1%"
                ),
                ProblemMetric(
                    "factual_accuracy", "Factual accuracy score", True, 1.2, ".1%"
                ),
                ProblemMetric(
                    "coherence_score", "Summary coherence and flow", True, 0.8, ".2f"
                ),
                ProblemMetric(
                    "conciseness_ratio", "Information density", True, 0.6, ".2f"
                ),
                ProblemMetric("readability_score", "Ease of reading", True, 0.4, ".2f"),
            ],
            optimization_objectives=["content_coverage"],
            expected_model_ranking=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        )

    def __init__(self, config: Optional[ProblemConfig] = None):
        if config is None:
            config = self.get_default_config()
        super().__init__(config)

    def create_dataset(self) -> Dataset:
        """Create challenging document summarization dataset."""
        examples_data = [
            # Technical Articles
            {
                "title": "Machine Learning Model Optimization Techniques",
                "content": """Machine learning model optimization is crucial for achieving high performance in real-world applications. Several key techniques have emerged as industry standards. First, hyperparameter tuning involves systematically searching for the best configuration of model parameters. Popular methods include grid search, random search, and Bayesian optimization. Grid search exhaustively tests all combinations within a defined parameter space, while random search samples randomly from the parameter distributions. Bayesian optimization uses probabilistic models to guide the search more efficiently. Second, feature engineering plays a vital role in model performance. This includes feature selection, which removes irrelevant or redundant features, and feature transformation, which creates new features from existing ones. Techniques like principal component analysis (PCA) can reduce dimensionality while preserving important information. Third, ensemble methods combine multiple models to achieve better performance than individual models. Popular ensemble techniques include bagging, boosting, and stacking. Random forests use bagging with decision trees, while gradient boosting machines iteratively improve predictions by learning from previous errors. Finally, regularization techniques like L1 and L2 regularization help prevent overfitting by adding penalty terms to the loss function.""",
                "expected_summary": "Machine learning optimization involves four main techniques: hyperparameter tuning (using grid search, random search, or Bayesian optimization), feature engineering (selection and transformation including PCA), ensemble methods (bagging, boosting, stacking like Random Forests and Gradient Boosting), and regularization (L1/L2) to prevent overfitting.",
                "difficulty": "medium",
                "domain": "technical",
            },
            {
                "title": "Climate Change Impact on Agricultural Systems",
                "content": """Climate change is fundamentally altering agricultural systems worldwide, with far-reaching implications for food security. Rising temperatures are shifting growing seasons and altering precipitation patterns, forcing farmers to adapt their practices. In many regions, traditional crops are becoming less viable as temperature and moisture conditions change beyond optimal ranges. Extreme weather events, including droughts, floods, and heat waves, are becoming more frequent and severe, causing significant crop losses and disrupting supply chains. Water scarcity is emerging as a critical challenge, particularly in arid and semi-arid regions where irrigation is essential. Changes in pest and disease pressure are also evident, as warmer temperatures and altered humidity levels create more favorable conditions for certain agricultural pests and pathogens. Some regions may experience benefits from longer growing seasons and increased atmospheric CO2 levels, which can enhance photosynthesis in certain crops. However, these potential benefits are often offset by negative impacts such as heat stress, water limitations, and soil degradation. Adaptation strategies include developing climate-resilient crop varieties, implementing precision agriculture technologies, improving water management systems, and diversifying cropping systems. International cooperation and knowledge sharing are essential for developing effective responses to these global challenges.""",
                "expected_summary": "Climate change is severely impacting agriculture through altered temperatures, precipitation, and extreme weather, leading to crop losses and water scarcity. While some regions may benefit from longer growing seasons and higher CO2, negative effects like heat stress and new pest pressures dominate. Adaptation requires climate-resilient crops, precision agriculture, better water management, and international cooperation.",
                "difficulty": "hard",
                "domain": "environmental",
            },
            # Business/Economic Content
            {
                "title": "Digital Transformation in Retail Industry",
                "content": """The retail industry is undergoing a massive digital transformation driven by changing consumer expectations and technological advances. E-commerce platforms have become essential, with many retailers adopting omnichannel strategies that seamlessly integrate online and offline experiences. Mobile commerce is particularly significant, as smartphones become the primary shopping device for many consumers. Artificial intelligence and machine learning are revolutionizing various aspects of retail operations. Personalization engines analyze customer data to provide tailored product recommendations and customized shopping experiences. Inventory management systems use predictive analytics to optimize stock levels and reduce waste. Customer service is being enhanced through chatbots and virtual assistants that can handle routine inquiries 24/7. Supply chain management is becoming more sophisticated with real-time tracking, automated warehouses, and drone delivery systems in pilot programs. Social commerce is emerging as platforms like Instagram and TikTok integrate shopping features directly into their social media experiences. Augmented reality applications allow customers to virtually try products before purchasing, reducing return rates and improving satisfaction. However, this transformation also presents challenges including cybersecurity risks, data privacy concerns, and the need for significant technology investments. Retailers must balance innovation with security and ensure that digital initiatives truly enhance rather than complicate the customer experience.""",
                "expected_summary": "Retail is transforming digitally through omnichannel strategies, mobile commerce, and AI-powered personalization and inventory management. Key innovations include chatbots, automated supply chains, social commerce, and AR try-on features. While offering enhanced customer experiences, retailers face challenges in cybersecurity, privacy, and balancing innovation with usability.",
                "difficulty": "medium",
                "domain": "business",
            },
            # Scientific/Medical Content
            {
                "title": "CRISPR Gene Editing: Therapeutic Applications and Ethical Considerations",
                "content": """CRISPR-Cas9 gene editing technology has emerged as one of the most promising tools for treating genetic diseases, offering unprecedented precision in modifying DNA sequences. The system works by using a guide RNA to direct the Cas9 enzyme to specific genomic locations where it creates double-strand breaks, allowing for targeted insertions, deletions, or replacements of genetic material. Clinical applications are advancing rapidly across multiple therapeutic areas. In ophthalmology, CRISPR treatments for inherited blindness caused by mutations in genes like CEP290 are showing promising results in early trials. Sickle cell disease and beta-thalassemia patients have been successfully treated using ex vivo gene editing of hematopoietic stem cells. Cancer immunotherapy approaches involve editing T-cells to enhance their ability to recognize and destroy tumor cells. Neurological applications are being explored for conditions like Huntington's disease and amyotrophic lateral sclerosis, though delivery to the central nervous system remains challenging. Despite tremendous therapeutic potential, CRISPR raises significant ethical questions. Germline editing, which affects inherited genetic changes, is particularly controversial as modifications could be passed to future generations. The 2018 announcement of gene-edited babies in China sparked international condemnation and calls for stricter oversight. Issues of accessibility and equity are also concerning, as these advanced therapies may initially be available only to wealthy patients in developed countries. International coordination is needed to establish appropriate regulatory frameworks and ethical guidelines while supporting continued research into this transformative technology.""",
                "expected_summary": "CRISPR-Cas9 gene editing enables precise DNA modification for treating genetic diseases, with clinical progress in inherited blindness, blood disorders, and cancer immunotherapy. However, germline editing raises ethical concerns about inherited changes, while accessibility issues and the need for international regulatory frameworks remain significant challenges alongside the technology's therapeutic promise.",
                "difficulty": "very_hard",
                "domain": "scientific",
            },
            # Historical/Cultural Content
            {
                "title": "The Silk Road: Ancient Trade Networks and Cultural Exchange",
                "content": """The Silk Road was not a single route but a vast network of trade paths connecting East and West for over 1,400 years, from approximately 130 BCE to 1453 CE. These routes facilitated not only commercial exchange but also profound cultural, technological, and intellectual interactions between civilizations. The network extended from Xi'an in China through Central Asia, the Middle East, and ultimately to Mediterranean ports like Constantinople and Venice. Silk was indeed a major trade commodity, prized in Rome for its luxury and beauty, but the routes carried much more diverse goods including spices, precious stones, metals, ceramics, and exotic animals. Perhaps more importantly, the Silk Road enabled the transmission of ideas, religions, technologies, and artistic styles. Buddhism spread from India to China and beyond through these networks. Islamic scholarship, including advances in mathematics, astronomy, and medicine, traveled westward. Chinese innovations like papermaking, printing, and gunpowder eventually reached Europe via these connections. The routes were maintained by various empires and city-states that profited from taxation and trade facilitation. Caravanserai - fortified roadside inns - provided shelter and security for merchants. The Mongol Empire's vast reach in the 13th and 14th centuries provided unprecedented security and facilitated trade. The decline of the Silk Road began with the rise of maritime trade routes and the fall of Constantinople to the Ottoman Empire, which disrupted traditional overland connections.""",
                "expected_summary": "The Silk Road was a 1,400-year network of trade routes connecting East and West, facilitating exchange of goods like silk, spices, and precious materials. More significantly, it enabled cultural transmission including Buddhism's spread, Islamic scholarship, and Chinese innovations like paper and gunpowder. Supported by empires and caravanserai, it declined with maritime trade and Ottoman control of Constantinople.",
                "difficulty": "hard",
                "domain": "historical",
            },
            # Complex Multi-Topic Content
            {
                "title": "Artificial Intelligence in Healthcare: Promises and Challenges",
                "content": """Artificial intelligence is poised to revolutionize healthcare delivery, diagnosis, and treatment, but implementation faces significant technical, ethical, and regulatory hurdles. Machine learning algorithms excel at pattern recognition in medical imaging, often achieving diagnostic accuracy comparable to or exceeding human specialists. Radiology applications include detecting cancer in mammograms, identifying diabetic retinopathy in eye scans, and analyzing chest X-rays for pneumonia. Pathology is being transformed by AI systems that can identify cancer cells in tissue samples with remarkable precision. Drug discovery is accelerating through AI-powered molecular design and virtual screening of compound libraries. Electronic health records provide vast datasets for predictive analytics, enabling early identification of patients at risk for complications or adverse events. However, several challenges impede widespread adoption. Data quality and interoperability remain major obstacles, as healthcare systems often use incompatible formats and standards. Algorithm bias is a serious concern, particularly when training data lacks diversity, potentially leading to disparate outcomes across different patient populations. Regulatory approval processes are still evolving to address AI-specific considerations. Privacy and security concerns are paramount given the sensitive nature of health information. The 'black box' nature of many AI algorithms raises questions about explainability and accountability in clinical decision-making. Healthcare professionals require extensive training to effectively integrate AI tools into their workflows. Despite these challenges, the potential benefits are enormous, including improved diagnostic accuracy, reduced healthcare costs, and enhanced access to specialized care in underserved areas.""",
                "expected_summary": "AI is transforming healthcare through superior medical imaging diagnosis, accelerated drug discovery, and predictive analytics from health records. Key challenges include data interoperability, algorithm bias, evolving regulations, privacy concerns, lack of explainability, and need for professional training. Despite obstacles, AI promises improved accuracy, reduced costs, and better access to specialized care.",
                "difficulty": "very_hard",
                "domain": "medical_tech",
            },
        ]

        # Add more examples to reach 20
        additional_examples = [
            {
                "title": "Renewable Energy Grid Integration Challenges",
                "content": """Integrating renewable energy sources into existing electrical grids presents complex technical and economic challenges that require innovative solutions. Solar and wind power are inherently intermittent, producing electricity only when environmental conditions are favorable. This variability creates difficulties for grid operators who must maintain constant balance between electricity supply and demand. Traditional power plants can adjust output relatively quickly, but renewable sources cannot be controlled in the same way. Energy storage systems, particularly large-scale battery installations, are becoming crucial for managing this intermittency by storing excess energy during peak production and releasing it during low production periods. Smart grid technologies enable better monitoring and control of distributed energy sources, allowing for more sophisticated demand response programs. Grid modernization requires significant infrastructure investments, including upgraded transmission lines capable of handling bidirectional power flows and advanced control systems. Economic challenges include the declining cost competitiveness of renewable energy, which is making traditional fossil fuel plants economically unviable in many markets. Policy frameworks must evolve to support this transition while ensuring grid reliability and affordability for consumers.""",
                "expected_summary": "Renewable energy integration faces challenges from solar/wind intermittency that disrupt grid supply-demand balance. Solutions include energy storage systems, smart grid technologies, infrastructure upgrades for bidirectional power flows, and policy frameworks supporting the transition while maintaining reliability and affordability.",
                "difficulty": "hard",
                "domain": "energy",
            },
            {
                "title": "Urban Planning and Sustainable City Development",
                "content": """Modern urban planning increasingly focuses on creating sustainable, resilient cities that can accommodate growing populations while minimizing environmental impact. Compact, mixed-use development reduces urban sprawl and transportation needs by integrating residential, commercial, and office spaces within walkable neighborhoods. Public transportation systems, including buses, trains, and emerging technologies like autonomous shuttles, are essential for reducing private vehicle dependence and associated emissions. Green infrastructure incorporates natural systems into urban design through features like green roofs, urban forests, rain gardens, and permeable surfaces that manage stormwater and improve air quality. Affordable housing strategies must balance market forces with social equity to prevent displacement of existing communities. Circular economy principles are being applied to urban waste management, with programs for recycling, composting, and waste-to-energy conversion. Smart city technologies use sensors and data analytics to optimize traffic flow, energy consumption, and public services. However, urban development faces challenges including funding constraints, regulatory complexities, and the need to retrofit existing infrastructure while accommodating rapid population growth.""",
                "expected_summary": "Sustainable urban planning emphasizes compact mixed-use development, public transportation, and green infrastructure to minimize environmental impact. Key strategies include affordable housing, circular waste management, and smart city technologies, though implementation faces funding, regulatory, and infrastructure retrofit challenges amid rapid population growth.",
                "difficulty": "medium",
                "domain": "urban_planning",
            },
        ]

        # Continue with shorter, varied examples to reach 20 total
        all_examples = examples_data + additional_examples

        # Add simpler examples for balance
        simple_examples = [
            {
                "title": "Benefits of Regular Exercise",
                "content": """Regular physical exercise provides numerous health benefits for both body and mind. Cardiovascular health improves through strengthened heart muscle and better blood circulation. Weight management becomes easier as exercise burns calories and builds muscle mass. Mental health benefits include reduced stress, anxiety, and depression through endorphin release. Sleep quality typically improves with regular physical activity. Bone density increases, reducing risk of osteoporosis. Exercise also boosts immune system function and increases energy levels throughout the day.""",
                "expected_summary": "Regular exercise improves cardiovascular health, aids weight management, enhances mental wellbeing through endorphin release, improves sleep quality, strengthens bones, boosts immunity, and increases daily energy levels.",
                "difficulty": "easy",
                "domain": "health",
            },
            {
                "title": "Introduction to Cloud Computing",
                "content": """Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, and software. The main service models are Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). Key benefits include cost savings through pay-as-you-use pricing, scalability to handle varying workloads, and accessibility from anywhere with internet connection. Major cloud providers include Amazon Web Services, Microsoft Azure, and Google Cloud Platform. Security and data privacy remain important considerations when moving to cloud services.""",
                "expected_summary": "Cloud computing provides internet-based computing services through IaaS, PaaS, and SaaS models. Benefits include cost savings, scalability, and global accessibility via providers like AWS, Azure, and Google Cloud, though security and privacy considerations remain important.",
                "difficulty": "easy",
                "domain": "technology",
            },
        ]

        # Combine all examples
        all_examples.extend(simple_examples)

        examples = []
        for i, data in enumerate(all_examples[:20]):  # Limit to 20 examples
            example = EvaluationExample(
                input_data={
                    "title": data["title"],
                    "content": data["content"],
                    "target_length": "2-3 sentences",
                },
                expected_output=data["expected_summary"],
                metadata={
                    "difficulty": data["difficulty"],
                    "domain": data["domain"],
                    "example_id": f"sum_{i+1:03d}",
                    "content_length": len(data["content"]),
                    "summary_length": len(data["expected_summary"]),
                },
            )
            examples.append(example)

        return Dataset(
            examples=examples,
            name="Document Summarization Quality",
            description=f"Document summarization with {len(examples)} examples across multiple domains and difficulty levels",
        )

    def create_function(self) -> Callable:
        """Create the base document summarizer function."""

        def document_summarizer(title: str, content: str, target_length: str) -> str:
            """Summarize documents maintaining key information and coherence."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=200,
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert document summarizer. Create concise, accurate summaries that capture the most important information while maintaining clarity and coherence.",
                    ),
                    (
                        "human",
                        """Summarize the following document:

Title: {title}
Content: {content}
Target Length: {target_length}

Provide a clear, accurate summary that captures the key points and main ideas.""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke(
                {"title": title, "content": content, "target_length": target_length}
            )["text"]

            return result.strip()

        return document_summarizer

    def create_optimized_function(self) -> Callable:
        """Create the optimized document summarizer."""

        @traigent.optimize(
            eval_dataset=self.create_temporary_dataset_file(),
            objectives=self.get_optimization_objectives(),
            configuration_space=self.get_configuration_space(),
            auto_override_frameworks=True,
            framework_targets=["langchain_openai.ChatOpenAI"],
            execution_mode="edge_analytics",
        )
        def document_summarizer_optimized(
            title: str, content: str, target_length: str
        ) -> str:
            """Optimized document summarizer."""
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Will be overridden by Traigent
                temperature=0.3,  # Will be overridden by Traigent
                max_tokens=200,  # Will be overridden by Traigent
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert document summarizer. Create concise, accurate summaries that capture the most important information while maintaining clarity and coherence.",
                    ),
                    (
                        "human",
                        """Summarize the following document:

Title: {title}
Content: {content}
Target Length: {target_length}

Provide a clear, accurate summary that captures the key points and main ideas.""",
                    ),
                ]
            )

            chain = LLMChain(llm=llm, prompt=prompt_template)
            result = chain.invoke(
                {"title": title, "content": content, "target_length": target_length}
            )["text"]

            return result.strip()

        return document_summarizer_optimized

    def evaluate_custom_metrics(
        self,
        outputs: List[Any],
        expected_outputs: List[Any],
        errors: List[Optional[str]],
    ) -> Dict[str, float]:
        """Compute document summarization specific metrics."""
        metrics = {}

        dataset = self.get_dataset()

        # Content coverage (simplified - measures length similarity as proxy)
        coverage_scores = []
        for output, expected, error in zip(outputs, expected_outputs, errors):
            if error is None and expected is not None and output:
                # Simple length-based coverage score
                output_words = len(str(output).split())
                expected_words = len(str(expected).split())
                if expected_words > 0:
                    coverage = min(output_words / expected_words, 1.0)
                    coverage_scores.append(coverage)

        metrics["content_coverage"] = (
            sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        )

        # Factual accuracy (simplified - uses keyword overlap)
        accuracy_scores = []
        for output, expected, error in zip(outputs, expected_outputs, errors):
            if error is None and expected is not None and output:
                output_words = set(str(output).lower().split())
                expected_words = set(str(expected).lower().split())
                if expected_words:
                    overlap = len(output_words.intersection(expected_words))
                    accuracy = overlap / len(expected_words)
                    accuracy_scores.append(accuracy)

        metrics["factual_accuracy"] = (
            sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        )

        # Coherence score (placeholder - could use more sophisticated NLP metrics)
        metrics["coherence_score"] = metrics["factual_accuracy"] * 0.9  # Simplified

        # Conciseness ratio (information density)
        conciseness_scores = []
        for i, (output, expected, error) in enumerate(
            zip(outputs, expected_outputs, errors)
        ):
            if (
                error is None
                and expected is not None
                and output
                and i < len(dataset.examples)
            ):
                content = dataset.examples[i].input_data.get("content", "")
                if content:
                    output_len = len(str(output))
                    content_len = len(content)
                    # Higher score for more concise summaries
                    conciseness = (
                        1.0 - (output_len / content_len) if content_len > 0 else 0.0
                    )
                    conciseness_scores.append(max(0.0, conciseness))

        metrics["conciseness_ratio"] = (
            sum(conciseness_scores) / len(conciseness_scores)
            if conciseness_scores
            else 0.0
        )

        # Readability score (simplified)
        metrics["readability_score"] = metrics["coherence_score"] * 0.95  # Placeholder

        return metrics


# Register this problem
register_problem("document_summarization", DocumentSummarizationProblem)
