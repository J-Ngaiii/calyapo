def pred_is_correct(llm_out: str, true_ans: str):
    llm_out = llm_out.strip().upper()
    true_ans = true_ans.strip().upper()
    valid_answers = {'A', 'B', 'C', 'D', 'E', 'A.', 'B.', 'C.', 'D.', 'E.'}

    if true_ans not in {'A', 'B', 'C', 'D', 'E'}:
        raise ValueError(f"Inputted true answer is not valid: {true_ans}")

    if not llm_out:
        return False

    llm_output_valid = llm_out in valid_answers
    llm_output_correct = llm_out[0] == true_ans
    return llm_output_valid and llm_output_correct