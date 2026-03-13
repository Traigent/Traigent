import { optimize, param } from '../../../dist/index.js';

async function answerQuestion(question) {
  const model = 'cheap';
  const temperature = 0.2;

  if (model === 'accurate' && temperature === 0) {
    return 'Paris';
  }

  return question.includes('capital') ? 'unknown' : 'Helpful answer';
}

export const optimizedAnswerQuestion = optimize({
  configurationSpace: {
    model: param.enum(['cheap', 'accurate']),
    temperature: param.float({ min: 0, max: 0.2, step: 0.2 }),
  },
  objectives: ['accuracy'],
})(answerQuestion);
