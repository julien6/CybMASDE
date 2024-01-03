import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DetailPanelComponent } from './detail-panel.component';

describe('DetailPanelComponent', () => {
  let component: DetailPanelComponent;
  let fixture: ComponentFixture<DetailPanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DetailPanelComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DetailPanelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
