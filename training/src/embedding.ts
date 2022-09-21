export default class Embedding {
	constructor(private readonly vocab: { [token: string]: number }, private readonly emb: ort.InferenceSession) { 

	}

	public async run(texts: string[], shape: number[]): Promise<ort.Tensor> {
		// TODO Given a string, tokenize, lookup the token ids, and return the token ids as a tensor.
		const tokens = texts.map(text => this.tokenize(text))
		const tokenIds = tokens.map(tokens => tokens.map(token => this.vocab[token]))
		const feeds = {
			input: new ort.Tensor('int64', new BigInt64Array(tokenIds), shape)
		}
		const runResult = await this.emb.run(feeds)
		return runResult['output']
	}

	public tokenize(text: string): string[] {
		return text.split(/\s+/)
	}
}