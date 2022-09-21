import { flatten } from './tensor-utils'

export default class Embedding {
	constructor(private readonly vocab: { [token: string]: number }, private readonly oov: number, private readonly emb: ort.InferenceSession) {
	}

	public getTokenId(token: string): number {
		const result = this.vocab[token]
		if (result !== undefined) {
			return result
		}
		return this.oov
	}

	public async forward(texts: string[] | string[][], maxNumTokens: number | undefined = undefined): Promise<ort.Tensor> {
		let feeds: ort.InferenceSession.OnnxValueMapType
		if (Array.isArray(texts[0])) {
			if (maxNumTokens === undefined) {
				throw new Error('`maxNumTokens` must be defined.')
			}
			const tokenIds = texts.map(texts => this.getTokenIds(texts as string[], maxNumTokens))
			const batchSize = tokenIds.length
			feeds = {
				input: new ort.Tensor('int64', flatten(tokenIds), [batchSize, texts[0].length, maxNumTokens])
			}
		} else {
			const tokenIds = this.getTokenIds(texts as string[], maxNumTokens)
			if (maxNumTokens === undefined) {
				maxNumTokens = Math.max(...tokenIds.map(t => t.length))
			}
			const batchSize = tokenIds.length
			feeds = {
				input: new ort.Tensor('int64', new BigInt64Array(flatten(tokenIds)), [batchSize, maxNumTokens])
			}
		}
		const runResult = await this.emb.run(feeds)
		return runResult['output']
	}

	private getTokenIds(texts: string[], maxNumTokens: number | undefined) {
		const tokens = texts.map(text => this.tokenize(text, maxNumTokens))
		return tokens.map(tokens => tokens.map(token => this.getTokenId(token)))
	}

	public tokenize(text: string, maxNumTokens: number | undefined = undefined): string[] {
		return text.trim().split(/\s+/, maxNumTokens)
	}
}