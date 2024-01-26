package main

import (
	rl "github.com/gen2brain/raylib-go/raylib"
)

func main() {
	rl.InitWindow(800, 450, "Neuro Cluster")
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)

	for !rl.WindowShouldClose() {
		rl.BeginDrawing()

		rl.ClearBackground(rl.Orange)
		rl.DrawText("Hello from Neuro Cluster!", 190, 200, 20, rl.White)

		rl.EndDrawing()
	}
}
