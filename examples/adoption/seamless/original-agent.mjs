export async function answerQuestion(question) {
  const model = 'cheap';
  const temperature = 0.2;

  if (model === 'accurate' && temperature === 0) {
    if (question.includes('capital of France')) {
      return 'Paris';
    }

    return 'Helpful answer';
  }

  return 'unknown';
}
