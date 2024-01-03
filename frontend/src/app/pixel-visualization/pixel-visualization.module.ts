import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PixelVisualizationComponent } from './pixel-visualization.component';



@NgModule({
  declarations: [
    PixelVisualizationComponent
  ],
  imports: [
    CommonModule
  ],
  exports: [
    PixelVisualizationComponent
  ]
})
export class PixelVisualizationModule { }
