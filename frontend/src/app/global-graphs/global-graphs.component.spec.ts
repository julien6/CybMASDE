import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GlobalGraphsComponent } from './global-graphs.component';

describe('GlobalGraphsComponent', () => {
  let component: GlobalGraphsComponent;
  let fixture: ComponentFixture<GlobalGraphsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GlobalGraphsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GlobalGraphsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
