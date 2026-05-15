package booking;

import booking.grpc.Booking;
import booking.grpc.BookingServiceGrpc;
import booking.grpc.DeleteBookingRequest;
import booking.grpc.DeleteBookingResponse;
import booking.grpc.ListBookingsRequest;
import booking.grpc.ListBookingsResponse;
import io.grpc.stub.StreamObserver;

public class BookingServiceImpl extends BookingServiceGrpc.BookingServiceImplBase {
    private static final int CAPACITY = 5;
    private final BookingRecord[] rows = new BookingRecord[CAPACITY];

    public BookingServiceImpl() {
        for (int i = 0; i < CAPACITY; i++) {
            rows[i] = new BookingRecord();
        }
        rows[0].setId("id");
        rows[0].setNumPeople(56);
        rows[0].setBookingType("piazzola deluxe");
        rows[0].setVehicle("niente");
        rows[0].setImageFile("ciao.txt");
    }

    @Override
    public void deleteBooking(DeleteBookingRequest request, StreamObserver<DeleteBookingResponse> responseObserver) {
        int result = BookingRecord.deleteById(rows, request.getId());
        responseObserver.onNext(DeleteBookingResponse.newBuilder().setResult(result).build());
        responseObserver.onCompleted();
    }

    @Override
    public void listBookings(ListBookingsRequest request, StreamObserver<ListBookingsResponse> responseObserver) {
        ListBookingsResponse.Builder builder = ListBookingsResponse.newBuilder();
        for (BookingRecord row : rows) {
            if (row.matches(request.getBookingType(), request.getMinPeople())) {
                builder.addBookings(toProto(row));
            }
        }
        responseObserver.onNext(builder.build());
        responseObserver.onCompleted();
    }

    private static Booking toProto(BookingRecord row) {
        return Booking.newBuilder()
                .setId(row.getId())
                .setNumPeople(row.getNumPeople())
                .setBookingType(row.getBookingType())
                .setVehicle(row.getVehicle())
                .setPlate(row.getPlate())
                .setImageFile(row.getImageFile())
                .build();
    }
}
