#!/usr/bin/env python3
"""
Retrieval Evaluation Script for Credit Risk Assessment
Tests precision and recall of chunk retrieval for credit risk questions.
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import mlflow
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

from langchain_rag_pipeline import LangChainRAGPipeline
from utilities.metrics_calculator import MetricsCalculator

mlflow.langchain.autolog()

def get_experiment_configs():
    """Default function to get experiment configurations. Can be overridden by run_experiments.py"""
    return [
        {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 10,
            "recontextualize": False,
            "include_title_in_content": False
        }
    ]

@dataclass
class EvaluationQuestion:
    """Evaluation question with ground truth chunks and answer."""
    question: str
    relevant_chunks: List[str]  
    answer: str  


@dataclass
class RAGEvaluationQuestion:
    """Evaluation question with ground truth answer for RAG chain evaluation."""
    question: str
    ground_truth_answer: str  


class RetrievalEvaluator:
    """Evaluates retrieval precision and recall for credit risk questions."""
    
    def __init__(self, pipeline: LangChainRAGPipeline, judge_model: str = "gpt-4.1"):
        self.pipeline = pipeline
        self.metrics_calculator = MetricsCalculator(judge_model='gpt-4.1')
        
    def create_evaluation_questions(self) -> List[EvaluationQuestion]:
        """Create evaluation questions with ground truth chunks for credit risk assessment."""
        
        json_file_path = Path(__file__).parent / "evaluation_questions.json"
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)

            evaluation_questions = []
            for question_data in questions_data:
                evaluation_questions.append(
                    EvaluationQuestion(
                        question=question_data["question"],
                        relevant_chunks=question_data["relevant_chunks"],
                        answer=question_data["answer"]
                    )
                )
            
            return evaluation_questions
            
        except Exception as e:
            raise ValueError(f"Error in create_evaluation_questions: {e}")
    
    def get_retrieved_chunks(self, question: str) -> List[str]:
        """Get chunks retrieved by the RAG pipeline for a given question."""

        vectorstore = self.pipeline.load_vectorstore()
        if vectorstore is None:
            raise ValueError("Vector store not found. Please run setup_pipeline first.")
        
        chunks = self.pipeline.get_source_chunks(vectorstore, question)
        return [chunk["text"] for chunk in chunks]
    
    def test_rag_chain(self, question: str) -> str:
        """Test the full RAG chain (retrieval + generation) with MLflow tracing."""
        rag_chain = self.pipeline.setup_pipeline(force_rebuild=False)
        
        response = rag_chain.invoke({"question": question})
        return response
    
    def create_rag_evaluation_questions(self) -> List[RAGEvaluationQuestion]:
        """Create evaluation questions with ground truth answers for RAG chain evaluation."""
        
        eval_questions = self.create_evaluation_questions()
        rag_eval_questions = []
        for eval_question in eval_questions:
            rag_eval_questions.append(
                RAGEvaluationQuestion(
                    question=eval_question.question,
                    ground_truth_answer=eval_question.answer
                )
            )
        
        return rag_eval_questions
    
    def evaluate_rag_chain_comprehensive(self, eval_questions: List[EvaluationQuestion]) -> Dict[str, Any]:
        """
        Evaluate RAG chain output quality using ROUGE-L, relevancy, and faithfulness metrics.
        
        Args:
            eval_questions: List of questions with ground truth answers and relevant chunks
            
        Returns:
            Dictionary containing evaluation results with ROUGE-L, relevancy, and faithfulness
        """
        print("Evaluating RAG Chain Output Quality with ROUGE-L, Relevancy, and Faithfulness...")
        print(f"DEBUG: Number of eval_questions: {len(eval_questions)}")
        
        results = {
            "evaluation_type": "rag_chain_comprehensive",
            "questions": [],
            "overall_metrics": {}
        }
        
        all_rougeL_f1 = []
        all_relevancy_scores = []
        all_faithfulness_scores = []
        
        for i, eval_question in enumerate(eval_questions):
            print(f"\nQuestion {i+1}: {eval_question.question}")           

            generated_answer = self.test_rag_chain(eval_question.question)
            print(f"Generated Answer: {generated_answer}")
            print(f"Ground Truth: {eval_question.answer}")
                
            retrieved_chunks = self.get_retrieved_chunks(eval_question.question)                
            rouge_scores = self.metrics_calculator.calculate_rouge_scores(generated_answer, eval_question.answer)
            relevancy_score = self.metrics_calculator.calculate_relevancy_score(eval_question.question, generated_answer)
            faithfulness_result = self.metrics_calculator.calculate_faithfulness_score_ragas(
                    question=eval_question.question,
                    contexts=retrieved_chunks,
                    answer=generated_answer,
                    ground_truths=[eval_question.answer]
                )
                
            all_rougeL_f1.append(rouge_scores["rougeL_f1"])
            all_relevancy_scores.append(relevancy_score)
            all_faithfulness_scores.append(faithfulness_result["faithfulness_score"])
                
            print(f"ROUGE-L F1: {rouge_scores['rougeL_f1']}")
            print(f"Relevancy Score: {relevancy_score}")
            print(f"Faithfulness Score: {faithfulness_result['faithfulness_score']}")
                
            question_result = {
                    "question": eval_question.question,
                    "generated_answer": generated_answer,
                    "ground_truth_answer": eval_question.answer,
                    "relevant_chunks": eval_question.relevant_chunks,
                    "retrieved_chunks": retrieved_chunks,
                    "rouge_scores": rouge_scores,
                    "relevancy_score": relevancy_score,
                    "faithfulness_score": faithfulness_result["faithfulness_score"],
                    "retrieval_scores": {
                        "precision": 0.0,  
                        "recall": 0.0,     
                        "ndcg": 0.0        
                    }
                }
                
            results["questions"].append(question_result)            

        rougeL_stats = self.metrics_calculator.calculate_overall_metrics(all_rougeL_f1)
        relevancy_stats = self.metrics_calculator.calculate_overall_metrics(all_relevancy_scores)
        faithfulness_stats = self.metrics_calculator.calculate_overall_metrics(all_faithfulness_scores)
        
        results["overall_metrics"] = {
            "mean_rougeL_f1": rougeL_stats["mean"],
            "mean_relevancy_score": relevancy_stats["mean"],
            "mean_faithfulness_score": faithfulness_stats["mean"],
            "std_rougeL_f1": rougeL_stats["std"],
            "std_relevancy_score": relevancy_stats["std"],
            "std_faithfulness_score": faithfulness_stats["std"]
        }        

        print(f"\nRAG Chain Comprehensive Evaluation Summary:")
        print(f"Mean ROUGE-L F1: {results['overall_metrics']['mean_rougeL_f1']:.3f}")
        print(f"Mean Relevancy Score: {results['overall_metrics']['mean_relevancy_score']:.3f}")
        print(f"Mean Faithfulness Score: {results['overall_metrics']['mean_faithfulness_score']:.3f}")
        print(f"Std ROUGE-L F1: {results['overall_metrics']['std_rougeL_f1']:.3f}")
        print(f"Std Relevancy Score: {results['overall_metrics']['std_relevancy_score']:.3f}")
        print(f"Std Faithfulness Score: {results['overall_metrics']['std_faithfulness_score']:.3f}")
        
        print("DEBUG: Returning results with keys:", results.keys())
        print("DEBUG: Number of questions:", len(results["questions"]))
        
        return results
    
    def evaluate_retrieval(self, eval_questions: List[EvaluationQuestion]) -> Dict[str, Any]:
        """Evaluate retrieval performance for all questions."""
        
        results = {
            "questions": [],
            "overall_metrics": {},
            "question_type_metrics": {}
        }
        
        all_precisions = []
        all_recalls = []
        all_ndcgs = []
        all_relevant_retrieved_counts = []
        
        for i, eval_q in enumerate(eval_questions):
            print(f"\nEvaluating Question {i+1}: {eval_q.question}")
            print("-" * 80)
            
            retrieved_chunks = self.get_retrieved_chunks(eval_q.question)

            precision, recall = self.metrics_calculator.calculate_precision_recall(retrieved_chunks, eval_q.relevant_chunks)
            ndcg = self.metrics_calculator.calculate_ndcg(retrieved_chunks, eval_q.relevant_chunks)
            
            relevant_retrieved_chunks = []
            for chunk in retrieved_chunks:
                if any(relevant_text.lower() in chunk.lower() for relevant_text in eval_q.relevant_chunks):
                    matched_texts = []
                    for relevant_text in eval_q.relevant_chunks:
                        if relevant_text.lower() in chunk.lower():
                            matched_texts.append(relevant_text)
                    
                    relevant_retrieved_chunks.append({
                        "retrieved_chunk": chunk,
                        "matched_relevant_texts": matched_texts
                    })
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_ndcgs.append(ndcg)
            all_relevant_retrieved_counts.append(len(relevant_retrieved_chunks))
            
            question_result = {
                "question": eval_q.question,
                "relevant_chunks": eval_q.relevant_chunks,
                "retrieved_chunks": retrieved_chunks,
                "relevant_retrieved_chunks": relevant_retrieved_chunks,
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "num_retrieved": len(retrieved_chunks),
                "num_relevant": len(eval_q.relevant_chunks),
                "num_relevant_retrieved": len(relevant_retrieved_chunks)
            }
            
            results["questions"].append(question_result)          

            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"nDCG: {ndcg:.3f}")
            print(f"Retrieved {len(retrieved_chunks)} chunks")
            print(f"Expected {len(eval_q.relevant_chunks)} relevant chunks")
            print(f"Relevant retrieved: {len(relevant_retrieved_chunks)} chunks")
            
            if relevant_retrieved_chunks:
                print("\nRelevant Retrieved Chunks:")
                for i, rel_chunk in enumerate(relevant_retrieved_chunks):
                    print(f"{i+1}. Matched texts: {len(rel_chunk['matched_relevant_texts'])}")
                    print(f"Chunk preview: {rel_chunk['retrieved_chunk'][:100]}...")
            else:
                print("\nNo relevant chunks found in retrieved chunks")
            
            print("\nRetrieved Chunks:")
            for j, chunk in enumerate(retrieved_chunks):
                print(f"  {j+1}. {chunk[:200]}...")
        
        precision_stats = self.metrics_calculator.calculate_overall_metrics(all_precisions)
        recall_stats = self.metrics_calculator.calculate_overall_metrics(all_recalls)
        ndcg_stats = self.metrics_calculator.calculate_overall_metrics(all_ndcgs)
        relevant_retrieved_stats = self.metrics_calculator.calculate_overall_metrics(all_relevant_retrieved_counts)
        
        results["overall_metrics"] = {
            "mean_precision": precision_stats["mean"],
            "mean_recall": recall_stats["mean"],
            "mean_ndcg": ndcg_stats["mean"],
            "mean_relevant_retrieved": relevant_retrieved_stats["mean"]
        }        
        
        return results
    
def main(judge_model: str = "gpt-4o-mini"):
    """Main evaluation function with MLflow logging."""
    
    load_dotenv()
    experiment_configs = get_experiment_configs()
    all_results_df = pd.DataFrame()
    mlflow.set_tracking_uri("http://localhost:5000")
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    experiment_name = "rag_parameter_optimization"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        print(f"Experiment '{experiment_name}' found with ID: {experiment.experiment_id}")
    else:
        print(f"Creating new experiment '{experiment_name}'")
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment with ID: {experiment_id}")
    
    for config in experiment_configs:
        recontext_status = "with recontextualization" if config.get('recontextualize', False) else "without recontextualization"
        title_status = "with title" if config.get('include_title_in_content', False) else "without title"
        print(f"Starting Run: k={config['top_k']}, chunk_size={config['chunk_size']}, chunk_overlap={config['chunk_overlap']} ({recontext_status}, {title_status})")
        
        recontext_str = "_recontext" if config.get('recontextualize', False) else ""
        title_str = "_title" if config.get('include_title_in_content', False) else ""
        with mlflow.start_run(run_name=f"k{config['top_k']}_chunk{config['chunk_size']}_overlap{config['chunk_overlap']}{recontext_str}{title_str}") as run:      

            mlflow.log_params({
                "evaluation_type": "retrieval_precision_recall",
                "pipeline_type": "langchain_rag",
                "question_domain": "credit_risk_assessment",
                "chunk_size": config['chunk_size'],
                "chunk_overlap": config['chunk_overlap'],
                "top_k": config['top_k'],
                "recontextualize": config.get('recontextualize', False),
                "include_title_in_content": config.get('include_title_in_content', False),
                "judge_model": judge_model
            })
            
            print(f"Starting Retrieval Evaluation for Credit Risk Assessment")
            print(f"Configuration: k={config['top_k']}, chunk_size={config['chunk_size']}, chunk_overlap={config['chunk_overlap']}, title_in_content={config.get('include_title_in_content', False)}")
            print("=" * 80)
            
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="vector_store_")
            
            pipeline = LangChainRAGPipeline(
                data_dir="../../data",  
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap'],
                persist_dir=temp_dir,  
                top_k=config['top_k'],
                recontextualize=config.get('recontextualize', False),
                include_title_in_content=config.get('include_title_in_content', False),
            )
            
            pipeline.clear_vectorstore()
            rag_chain = pipeline.setup_pipeline(force_rebuild=True)
            vectorstore_info = pipeline.get_vectorstore_info()
            mlflow.log_params({
                "vectorstore_status": vectorstore_info["status"],
                "vectorstore_count": vectorstore_info["count"],
                "vectorstore_directory": vectorstore_info["persist_directory"]
            })
            print(f"Vector store info: {vectorstore_info}")
            
            evaluator = RetrievalEvaluator(pipeline, judge_model=judge_model)
            eval_questions = evaluator.create_evaluation_questions()
            mlflow.log_param("num_questions", len(eval_questions))
            
            results = evaluator.evaluate_retrieval(eval_questions)
            
            comprehensive_results = evaluator.evaluate_rag_chain_comprehensive(eval_questions)
            print("comprehensive_results keys:", comprehensive_results.keys())
            print("comprehensive_results:", comprehensive_results)
            
            mlflow.log_metrics({
                "overall_mean_precision": results["overall_metrics"]["mean_precision"],
                "overall_mean_recall": results["overall_metrics"]["mean_recall"],
                "overall_mean_ndcg": results["overall_metrics"]["mean_ndcg"],
                "overall_mean_relevant_retrieved": results["overall_metrics"]["mean_relevant_retrieved"]
            })
                
            for i, question_result in enumerate(results["questions"]):
                mlflow.log_metrics({
                    f"q{i+1}_precision": question_result["precision"],
                    f"q{i+1}_recall": question_result["recall"],
                    f"q{i+1}_ndcg": question_result["ndcg"],
                    f"q{i+1}_relevant_retrieved": question_result["num_relevant_retrieved"]
                })
            
            for i, comprehensive_result in enumerate(comprehensive_results["questions"]):
                mlflow.log_metrics({
                    f"q{i+1}_rougeL_f1": comprehensive_result["rouge_scores"]["rougeL_f1"],
                    f"q{i+1}_rougeL_precision": comprehensive_result["rouge_scores"]["rougeL_precision"],
                    f"q{i+1}_rougeL_recall": comprehensive_result["rouge_scores"]["rougeL_recall"],
                    f"q{i+1}_relevancy_score": comprehensive_result["relevancy_score"],
                    f"q{i+1}_faithfulness_score": comprehensive_result["faithfulness_score"]
                })
            
            mlflow.log_metrics({
                "rougeL_mean_f1": comprehensive_results["overall_metrics"]["mean_rougeL_f1"],
                "relevancy_mean_score": comprehensive_results["overall_metrics"]["mean_relevancy_score"],
                "faithfulness_mean_score": comprehensive_results["overall_metrics"]["mean_faithfulness_score"],
                "rougeL_std_f1": comprehensive_results["overall_metrics"]["std_rougeL_f1"],
                "relevancy_std_score": comprehensive_results["overall_metrics"]["std_relevancy_score"],
                "faithfulness_std_score": comprehensive_results["overall_metrics"]["std_faithfulness_score"]
            })
            
            comprehensive_results["run_info"] = {
                "run_id": run.info.run_id,
                "parameters": {
                    "chunk_size": config['chunk_size'],
                    "chunk_overlap": config['chunk_overlap'],
                    "top_k": config['top_k'],
                    "recontextualize": config.get('recontextualize', False)
                },
                "vectorstore_info": vectorstore_info
            }
            
            comprehensive_results["overall_metrics"]["retrieval"] = {
                "mean_precision": results["overall_metrics"]["mean_precision"],
                "mean_recall": results["overall_metrics"]["mean_recall"],
                "mean_ndcg": results["overall_metrics"]["mean_ndcg"],
                "mean_relevant_retrieved": results["overall_metrics"]["mean_relevant_retrieved"]
            }
            
            retrieval_lookup = {retrieval_result["question"]: retrieval_result for retrieval_result in results["questions"]}
            
            for comprehensive_result in comprehensive_results["questions"]:
                if comprehensive_result["question"] in retrieval_lookup:
                    matching_retrieval_result = retrieval_lookup[comprehensive_result["question"]]
                    comprehensive_result["retrieval_scores"]["precision"] = matching_retrieval_result["precision"]
                    comprehensive_result["retrieval_scores"]["recall"] = matching_retrieval_result["recall"]
                    comprehensive_result["retrieval_scores"]["ndcg"] = matching_retrieval_result["ndcg"]
                else:
                    print(f"Warning: No matching retrieval result found for question: {comprehensive_result['question']}")
            
            print("comprehensive_results keys:", comprehensive_results.keys())
            print("comprehensive_results questions:", len(comprehensive_results["questions"]))
            
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            recontext_suffix = "_recontext" if config.get('recontextualize', False) else ""
            comprehensive_results_file = results_dir / f"comprehensive_evaluation_results_k{config['top_k']}_chunk{config['chunk_size']}{recontext_suffix}.json"
            
            with open(comprehensive_results_file, "w") as f:
                json.dump(comprehensive_results, f, indent=2)
            mlflow.log_artifact(str(comprehensive_results_file))
            
            # Save vector database results with retrieval information for first question
            if eval_questions:
                first_question = eval_questions[0].question
                vector_db_results_file = results_dir / f"vector_db_results_{pipeline.get_configuration_name()}_with_retrieval.csv"
                pipeline.save_vector_db_results(str(vector_db_results_file), first_question)
                mlflow.log_artifact(str(vector_db_results_file))
                
            print("\n" + "=" * 80)
            print(f"EVALUATION SUMMARY - k={config['top_k']}, chunk_size={config['chunk_size']}")
            print("=" * 80)
            print(f"Overall Mean Precision: {results['overall_metrics']['mean_precision']:.3f}")
            print(f"Overall Mean Recall: {results['overall_metrics']['mean_recall']:.3f}")
            print(f"Overall Mean nDCG: {results['overall_metrics']['mean_ndcg']:.3f}")
            print(f"Overall Mean Relevant Retrieved: {results['overall_metrics']['mean_relevant_retrieved']:.1f}")
            
            print(f"\nComprehensive RAG Chain Evaluation Summary:")
            print(f"Mean ROUGE-L F1: {comprehensive_results['overall_metrics']['mean_rougeL_f1']:.3f}")
            print(f"Mean Relevancy Score: {comprehensive_results['overall_metrics']['mean_relevancy_score']:.3f}")
            print(f"Mean Faithfulness Score: {comprehensive_results['overall_metrics']['mean_faithfulness_score']:.3f}")
            print(f"Std ROUGE-L F1: {comprehensive_results['overall_metrics']['std_rougeL_f1']:.3f}")
            print(f"Std Relevancy Score: {comprehensive_results['overall_metrics']['std_relevancy_score']:.3f}")
            print(f"Std Faithfulness Score: {comprehensive_results['overall_metrics']['std_faithfulness_score']:.3f}")
            print(f"Comprehensive RAG chain evaluation completed")
            
            print(f"\nRun completed! Run ID: {run.info.run_id}")
            print(f"Results saved to MLflow experiment: rag_parameter_optimization")
            print(f"Comprehensive results file: {comprehensive_results_file}")
            
            experiment_result = {
                'chunk_size': config['chunk_size'],
                'chunk_overlap': config['chunk_overlap'],
                'top_k': config['top_k'],
                'recontextualize': config.get('recontextualize', False),
                'include_title_in_content': config.get('include_title_in_content', False),
                'run_id': run.info.run_id, 
                'mean_precision': results['overall_metrics']['mean_precision'],
                'mean_recall': results['overall_metrics']['mean_recall'],
                'mean_ndcg': results['overall_metrics']['mean_ndcg'],
                'mean_relevant_retrieved': results['overall_metrics']['mean_relevant_retrieved'],
                'mean_rougeL_f1': comprehensive_results['overall_metrics']['mean_rougeL_f1'],
                'mean_relevancy_score': comprehensive_results['overall_metrics']['mean_relevancy_score'],
                'mean_faithfulness_score': comprehensive_results['overall_metrics']['mean_faithfulness_score'],
                'std_rougeL_f1': comprehensive_results['overall_metrics']['std_rougeL_f1'],
                'std_relevancy_score': comprehensive_results['overall_metrics']['std_relevancy_score'],
                'std_faithfulness_score': comprehensive_results['overall_metrics']['std_faithfulness_score'],
                'vectorstore_count': vectorstore_info['count'],
                'num_questions': len(eval_questions)
            }
            
            all_results_df = pd.concat([all_results_df, pd.DataFrame([experiment_result])], ignore_index=True)
    
    print(f"\n{'='*80}")
    print("ALL RUNS COMPLETED")
    print(f"{'='*80}")
    print("Runs completed:")
    for config in experiment_configs:
        print(f"- k={config['top_k']}, chunk_size={config['chunk_size']}, chunk_overlap={config['chunk_overlap']}")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print("all_results_df:", results_dir)
    df_file = results_dir / "all_experiment_results.csv"
    all_results_df.to_csv(df_file, index=False)  


if __name__ == "__main__":
    import sys
    judge_model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4.1"
    print(f"Using judge model: {judge_model}")
    main(judge_model=judge_model) 