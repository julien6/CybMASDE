import {
  Component,
  ViewChild,
  ElementRef,
  OnInit,
} from '@angular/core';
import mermaid from 'mermaid';

@Component({
  selector: 'app-global-seqdiagrams',
  templateUrl: './global-seqdiagrams.component.html',
  styleUrls: ['./global-seqdiagrams.component.css']
})
export class GlobalSeqdiagramsComponent implements OnInit {

  @ViewChild('mermaidDiv', { static: false }) mermaidDiv: ElementRef;

  public ngAfterViewInit(): void {

    const element: any = this.mermaidDiv.nativeElement;

    // TODO: dynamic sequence diagram
    const graphDefinition = `sequenceDiagram
    box Purple Alice & John
    participant A
    participant J
    end
    box Another Group
    participant B
    participant C
    end
    A->>J: Hello John, how are you?
    J->>A: Great!
    A->>B: Hello Bob, how is Charly ?
    B->>C: Hello Charly, how are you?
    `;
    mermaid.render('graphDiv', graphDefinition, (svgCode, bindFunctions: any) => {
      element.innerHTML = svgCode;
      if (bindFunctions !== undefined) {
        bindFunctions(element);
      }
    });
  }

  ngOnInit(): void {
    mermaid.initialize({
      securityLevel: 'loose'
    });
    mermaid.init();
  }
}