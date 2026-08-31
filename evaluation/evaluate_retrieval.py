from evaluation.retrieval_cases import RETRIEVAL_TEST_CASES



def evaluate_retrieval(retrieval_pipeline):

    correct = 0

    total = len(RETRIEVAL_TEST_CASES)


    for test_case in RETRIEVAL_TEST_CASES:
        query =test_case["query"]

        expected_file = test_case["expected_file"].replace("\\","/")

        retrieved_results = retrieval_pipeline.retrieve(query)

        retrieved_files = [document.metadata["file_path"].replace("\\","/") for document , score in retrieved_results]


        # Check if the expected file is in the retrieved results

        success = expected_file in retrieved_files

        if success:
            correct += 1


        print(f"Query: {query}")

        print(f"Expected File: {expected_file}")    

        print(f"Retrieved Files: {retrieved_files}")

        print("Pass" if success else "Fail")

        print("--------------------")


    accuracy = correct / total * 100
    print(f"Accuracy : {accuracy:.2f}")


    return accuracy    












