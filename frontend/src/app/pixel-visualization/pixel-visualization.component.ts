import { Input, OnChanges, OnInit, SimpleChanges } from '@angular/core';
import { Component, AfterViewInit, ViewChild, ElementRef } from '@angular/core';

@Component({
  selector: 'app-pixel-visualization',
  templateUrl: './pixel-visualization.component.html',
  styleUrls: ['./pixel-visualization.component.css']
})
export class PixelVisualizationComponent implements OnInit, AfterViewInit {

  @ViewChild('pixelCanvas', { static: true }) pixelCanvas: ElementRef<HTMLCanvasElement>;

  @Input() width: number;
  @Input() height: number;

  private _data: any;

  @Input() set data(value: string) {
    this._data = JSON.parse(value);
    this.generatePicture()
  }

  public constructor() {
  }

  public ngOnInit(): void {
  }

  get data(): any {
    return this._data;
  }

  generateColor(percentage: number): number[] {

    const rCol = (x: number) => {
      if (0 <= x && x < 1) {
        return 1;
      }
      if (1 <= x && x < 2) {
        return 2 - x;
      }
      if (2 <= x && x < 4) {
        return 0;
      }
      if (4 <= x && x < 5) {
        return x - 4;
      }
      if (5 <= x && x <= 6) {
        return 1;
      }
      return 0;
    }

    const gCol = (x: number) => {
      if (0 <= x && x < 1) {
        return x;
      }
      if (1 <= x && x < 3) {
        return 1;
      }
      if (3 <= x && x < 4) {
        return 4 - x;
      }
      if (4 <= x && x <= 6) {
        return 0;
      }
      return 0;
    }

    const bCol = (x: number) => {
      if (0 <= x && x < 2) {
        return 0;
      }
      if (2 <= x && x < 3) {
        return x - 2;
      }
      if (3 <= x && x < 5) {
        return 1;
      }
      if (5 <= x && x <= 6) {
        return 6 - x;
      }
      return 0;
    }

    return [Math.floor(rCol(percentage * 6) * 255),
    Math.floor(gCol(percentage * 6) * 255),
    Math.floor(bCol(percentage * 6) * 255)]

  }

  ngAfterViewInit() {
    this.generatePicture();
  }

  generatePicture() {
    const canvas = this.pixelCanvas.nativeElement;

    canvas.width = this.width;
    canvas.height = this.height;

    const ctx: any = canvas.getContext('2d');
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const pixels = imageData.data;

    // console.log("pixels length: ", pixels.length)
    // console.log("canvas width: ", canvas.width)
    // console.log("canvas height: ", canvas.height)

    for (let i = 0; i < pixels.length; i += 4) {

      const y = Math.floor(Math.floor(i / 4) / canvas.width);

      if (y == Math.floor(canvas.height / 2)) {
        pixels[i] = 0;        // R (red)
        pixels[i + 1] = 0;      // G (green)
        pixels[i + 2] = 0;      // B (blue)
        pixels[i + 3] = 255;    // A (alpha)
      } else {
        pixels[i] = 255;        // R (red)
        pixels[i + 1] = 255;      // G (green)
        pixels[i + 2] = 255;      // B (blue)
        pixels[i + 3] = 255;    // A (alpha)
      }

    }

    for (let i = 0; i < Object.keys(this.data).length; i += 1) {

      const xStart = i * 4;
      const yStart = Math.floor(canvas.height / 2);

      for (let observation of this.data[i].observations) {

        const yOffset = (yStart - 1 - observation) * 4 * canvas.width;

        const colors = this.generateColor(observation / Math.floor(this.height / 2));

        pixels[xStart + yOffset] = colors[0];
        pixels[xStart + yOffset + 1] = colors[1];
        pixels[xStart + yOffset + 2] = colors[2];
        pixels[xStart + yOffset + 3] = 255;
      }

      for (let action of this.data[i].actions) {

        const yOffset = (yStart + 1 + action) * 4 * canvas.width;

        const colors = this.generateColor(action / Math.floor(this.height / 2));

        pixels[xStart + yOffset] = colors[0];
        pixels[xStart + yOffset + 1] = colors[1];
        pixels[xStart + yOffset + 2] = colors[2];
        pixels[xStart + yOffset + 3] = 255;
      }

    }

    ctx.putImageData(imageData, 0, 0);

    this.pixelCanvas.nativeElement.style.width = "100%";
    this.pixelCanvas.nativeElement.style.height = "100%";
  }

}