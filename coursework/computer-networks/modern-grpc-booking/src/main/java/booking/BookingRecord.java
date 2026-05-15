package booking;

import java.io.File;

/**
 * In-memory booking row (same fields as legacy {@code Prenotazione}).
 */
public final class BookingRecord {
    private static final String EMPTY = "L";

    private String id = EMPTY;
    private int numPeople = -1;
    private String bookingType = EMPTY;
    private String vehicle = EMPTY;
    private String plate = EMPTY;
    private String imageFile = EMPTY;

    public String getId() {
        return id;
    }

    public int getNumPeople() {
        return numPeople;
    }

    public String getBookingType() {
        return bookingType;
    }

    public String getVehicle() {
        return vehicle;
    }

    public String getPlate() {
        return plate;
    }

    public String getImageFile() {
        return imageFile;
    }

    public synchronized void reset() {
        id = EMPTY;
        numPeople = -1;
        bookingType = EMPTY;
        vehicle = EMPTY;
        plate = EMPTY;
        imageFile = EMPTY;
    }

    public synchronized void setId(String id) {
        this.id = id;
    }

    public synchronized void setNumPeople(int numPeople) {
        this.numPeople = numPeople;
    }

    public synchronized void setBookingType(String bookingType) {
        this.bookingType = bookingType;
    }

    public synchronized void setVehicle(String vehicle) {
        this.vehicle = vehicle;
    }

    public synchronized void setImageFile(String imageFile) {
        this.imageFile = imageFile;
    }

    public boolean matches(String type, int minPeople) {
        return bookingType.equals(type) && numPeople > minPeople;
    }

    /**
     * @return 0 ok, 1 image delete error, 2 id not found
     */
    public static int deleteById(BookingRecord[] rows, String id) {
        int result = 2;
        for (BookingRecord row : rows) {
            if (!row.getId().equals(id)) {
                continue;
            }
            result = 1;
            File image = new File(row.getImageFile());
            if (image.delete()) {
                result = 0;
            }
            row.reset();
        }
        return result;
    }

    @Override
    public String toString() {
        return id + "  " + numPeople + "  " + bookingType + "  "
                + vehicle + "  " + plate + "  " + imageFile;
    }
}
