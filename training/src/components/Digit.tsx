import React from "react";

const sizeScale = 2

export interface Props {
	pixels: number[][]
	label: number
}

export function Digit(props: Props) {
	const canvasRef = React.useRef<HTMLCanvasElement>(null)
	const { pixels, label } = props

	React.useEffect(() => {
		const canvas = canvasRef.current
		if (canvas) {
			const ctx = canvas.getContext("2d")
			if (ctx) {
				for (let i = 0; i < pixels.length; i++) {
					for (let j = 0; j < pixels[i].length; j++) {
						ctx.fillStyle = `rgb(${pixels[i][j]},${pixels[i][j]},${pixels[i][j]})`
						ctx.fillRect(j * sizeScale, i * sizeScale, sizeScale, sizeScale)
					}
				}
			}
		}
	}, [])

	return (<>
		<canvas ref={canvasRef} width={sizeScale * pixels.length} height={sizeScale * pixels[0].length}></canvas>
		{/* {label} */}
	</>);
}