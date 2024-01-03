import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PixelVisualizationComponent } from './pixel-visualization.component';

describe('PixelVisualizationComponent', () => {
  let component: PixelVisualizationComponent;
  let fixture: ComponentFixture<PixelVisualizationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ PixelVisualizationComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PixelVisualizationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
