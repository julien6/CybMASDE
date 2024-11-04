import { Directive, ElementRef, Input, OnChanges, Renderer2 } from '@angular/core';

@Directive({
  selector: '[matTooltipHtml]'
})
export class MatTooltipHtmlDirective implements OnChanges {
  @Input() matTooltipHtml!: string;

  constructor(private el: ElementRef, private renderer: Renderer2) {}

  ngOnChanges() {
    if (this.matTooltipHtml) {
      this.renderer.setProperty(this.el.nativeElement, 'innerHTML', this.matTooltipHtml);
    }
  }
}
